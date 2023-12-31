import tensorflow as tf
from generator.gpt2.src import model

tf.compat.v1.disable_eager_execution()

def penalize_used(logits, output):

    # I want to change the indices of logits wherever the index is found in output
    change_tensor = tf.zeros_like(logits, dtype=logits.dtype)
    unique = tf.unique(output[0])[0]
    ones = tf.ones_like(unique, dtype=unique.dtype)
    indices = tf.expand_dims(unique, 1)

    updates = tf.scatter_nd(indices, ones, [logits.shape[1]])

    bool_tensor = tf.expand_dims(tf.cast(updates, tf.bool), 0)

    return tf.compat.v1.where(bool_tensor, logits * 0.85, logits)


def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.compat.v1.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )

    return tf.cond(tf.equal(k, 0), lambda: logits, lambda: _top_k(),)


def top_p_logits(logits, p):
    """Nucleus sampling"""
    batch, num = logits.shape.as_list()
    sorted_logits = tf.sort(logits, direction="DESCENDING", axis=-1)
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    indices = tf.stack(
        [
            tf.range(0, batch),
            # number of indices to include
            tf.minimum(tf.reduce_sum(tf.cast(cumulative_probs < p, tf.int32), axis=-1), num),
        ],
        axis=-1,
    )
    min_values = tf.gather_nd(sorted_logits, indices)
    return tf.compat.v1.where(logits < min_values, tf.ones_like(logits) * -1e10, logits,)


def sample_sequence(
    *,
    hparams,
    length,
    start_token=None,
    batch_size=None,
    context=None,
    temperature=1,
    top_p=1,
):
    if start_token is None:
        assert context is not None, "Specify exactly one of start_token and context!"
    else:
        assert context is None, "Specify exactly one of start_token and context!"
        context = tf.fill([batch_size, 1], start_token)

    def step(hparams, tokens, past=None):
        tokens = tf.convert_to_tensor(tokens)
        lm_output = model.model(
            hparams=hparams, X=tokens, past=past, reuse=tf.compat.v1.AUTO_REUSE
        )

        logits = lm_output["logits"][:, :, : hparams.n_vocab]
        presents = lm_output["present"]
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            "logits": logits,
            "presents": presents,
        }

    with tf.compat.v1.name_scope("sample_sequence"):
        def body(past, prev, output):
            next_outputs = step(hparams, prev, past=past)
            logits = next_outputs["logits"][:, -1, :] / tf.cast(temperature, dtype=tf.float32)
            logits = penalize_used(logits, output)
            logits = top_p_logits(logits, p=top_p)
            samples = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)
            return [
                tf.convert_to_tensor(next_outputs["presents"])
                if past is None
                else tf.convert_to_tensor(tf.concat([past, next_outputs["presents"]], axis=-2)),
                tf.convert_to_tensor(samples),
                tf.concat([tf.convert_to_tensor(output), tf.convert_to_tensor(samples)], axis=1),
            ]

        past, prev, output = body(None, context, context)

        def cond(*args):
            return True

        _, _, tokens = tf.while_loop(
            cond=cond,
            body=body,
            maximum_iterations=length - 1,
            loop_vars=[past, prev, output],
            shape_invariants=[
                tf.TensorShape(
                    model.past_shape(hparams=hparams, batch_size=batch_size)
                ),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),
            ],
            back_prop=False,
        )

        return tokens
