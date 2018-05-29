from cmput659.td import Td0, TdNstep
import tensorflow as tf
from tensorflow.python.layers.core import Dense as DenseLayer
import numpy as np


class TdTest(tf.test.TestCase):
    def test_td0_linear_single_output(self):
        with self.test_session() as sess:
            num_dimensions = 2
            num_players = 1
            num_examples = 10

            tf.set_random_seed(42)

            x = sess.run(
                tf.concat(
                    [
                        tf.random_normal(
                            [num_examples + 1, num_dimensions - 1]),
                        tf.ones([num_examples + 1, 1])
                    ],
                    axis=1))
            rewards = sess.run(tf.random_normal([num_examples, num_players]))

            x_placeholder = tf.placeholder(tf.float32, [None, num_dimensions])
            reward_placeholder = tf.placeholder(tf.float32, [None, 1])

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

            layer = DenseLayer(
                num_players,
                kernel_initializer=tf.zeros_initializer,
                use_bias=False)

            patient = Td0(x_placeholder, layer, optimizer, reward_placeholder)

            actual_loss = tf.losses.mean_squared_error(
                np.cumsum(rewards, axis=0), patient.evaluation_node())

            sess.run(tf.global_variables_initializer())

            td_zero_loss = sess.run(
                actual_loss, feed_dict={
                    patient.observations_node(): x[:-1]
                })
            self.assertAlmostEqual(1.4945195, td_zero_loss, places=7)
            self.assertEqual(1, sess.run(patient.sparsity_node))

            update_node = patient.create_learning_step_node()

            for i in range(10):
                for t in range(num_examples):
                    sess.run(
                        update_node,
                        feed_dict={
                            patient.observations_node(): x[t:t + 2, :],
                            patient.reward_node: rewards[t, :].reshape([1, 1])
                        })
                td_zero_loss = sess.run(
                    actual_loss,
                    feed_dict={
                        patient.observations_node(): x[:-1]
                    })
                self.assertLess(td_zero_loss, 1.4945195)
            self.assertAlmostEqual(1.4552972, td_zero_loss, places=7)
            self.assertEqual(0, sess.run(patient.sparsity_node))

    def test_td_nstep_linear_single_output(self):
        with self.test_session() as sess:
            num_dimensions = 2
            num_players = 1
            num_examples = 10

            tf.set_random_seed(42)

            x = sess.run(
                tf.concat(
                    [
                        tf.random_normal(
                            [num_examples + 1, num_dimensions - 1]),
                        tf.ones([num_examples + 1, 1])
                    ],
                    axis=1))
            rewards = sess.run(tf.random_normal([num_examples, num_players]))

            x_placeholder = tf.placeholder(tf.float32, [None, num_dimensions])
            reward_placeholder = tf.placeholder(tf.float32, [None, 1])

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

            layer = DenseLayer(
                num_players,
                kernel_initializer=tf.zeros_initializer,
                use_bias=False)

            patient = TdNstep(x_placeholder, layer, optimizer,
                              reward_placeholder)

            actual_loss = tf.losses.mean_squared_error(
                np.cumsum(rewards, axis=0), patient.evaluation_node())

            sess.run(tf.global_variables_initializer())

            td_n_loss = sess.run(
                actual_loss, feed_dict={
                    patient.observations_node(): x[:-1]
                })
            self.assertAlmostEqual(1.4945195, td_n_loss, places=7)
            self.assertEqual(1, sess.run(patient.sparsity_node))

            update_node = patient.create_learning_step_node()

            for i in range(10):
                sess.run(
                    update_node,
                    feed_dict={
                        patient.observations_node(): x,
                        patient.reward_node: rewards
                    })
                td_n_loss = sess.run(
                    actual_loss,
                    feed_dict={
                        patient.observations_node(): x[:-1]
                    })
                self.assertLess(td_n_loss, 1.4945195)
            self.assertAlmostEqual(1.468768, td_n_loss, places=6)
            self.assertEqual(0, sess.run(patient.sparsity_node))


if __name__ == '__main__':
    tf.test.main()
