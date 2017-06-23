import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from skimage.io import imsave
import os
import shutil
import time

img_height = 28
img_width = 28
img_size = img_height * img_width

to_train = True
to_restore = False
output_path = "output"

max_epoch = 1000

h1_size = 150
h2_size = 300
z_size = 100
batch_size = 256
seq_size=4
n_hidden=300
tr_data_num=60000;
g_num_layers=2;
d_num_layers=2;

log_dir="/tmp/gan_seq/"+str(int(time.time()))


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    #tf.summary.histogram('histogram', var)

def build_generator(z_prior,keep_prob):
    z_prior=tf.unstack(z_prior,seq_size,1);
    lstm_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(n_hidden), output_keep_prob=keep_prob)for _ in range(g_num_layers)]);
    with tf.variable_scope("gen") as gen:
      res, states = tf.contrib.rnn.static_rnn(lstm_cell, z_prior,dtype=tf.float32);
      weights=tf.Variable(tf.random_normal([n_hidden, img_size]));
      biases=tf.Variable(tf.random_normal([img_size]));
      for i in range(len(res)):
        res[i]=tf.nn.tanh(tf.matmul(res[i], weights) + biases);
      g_params=[v for v in tf.global_variables() if v.name.startswith(gen.name)];
    with tf.name_scope("gen_params"):
      for param in g_params:
        variable_summaries(param);
    return res,g_params;

def build_discriminator(x_data, x_generated, keep_prob):
    x_data=tf.unstack(x_data,seq_size,1);
    x_generated=list(x_generated);
    x_in = tf.concat([x_data, x_generated],1);
    x_in=tf.unstack(x_in,seq_size,0);
    lstm_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(n_hidden), output_keep_prob=keep_prob) for _ in range(d_num_layers)]);
    with tf.variable_scope("dis") as dis:
      weights=tf.Variable(tf.random_normal([n_hidden, 1]));
      biases=tf.Variable(tf.random_normal([1]));
      outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x_in, dtype=tf.float32);
      res=tf.matmul(outputs[-1], weights) + biases;
      y_data = tf.nn.sigmoid(tf.slice(res, [0, 0], [batch_size, -1], name=None));
      y_generated = tf.nn.sigmoid(tf.slice(res, [batch_size, 0], [-1, -1], name=None));
      d_params=[v for v in tf.global_variables() if v.name.startswith(dis.name)];
    with tf.name_scope("desc_params"):
      for param in d_params:
        variable_summaries(param);
    return y_data, y_generated, d_params;

def show_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], img_height, img_width)) + 0.5
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    imsave(fname, img_grid)


def train():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    total_tr_data, total_tr_label = mnist.train.next_batch(mnist.train._num_examples);
    total_tr_data=np.array(total_tr_data,dtype=float);
    total_tr_label=np.array(total_tr_label,dtype=float);
    
    tr_data=np.zeros((tr_data_num,seq_size,img_size),dtype=object);
    for i in range(seq_size):
      total_idx=np.where(total_tr_label[:,i]==1.0)[0];
      while(len(total_idx)<tr_data_num):
        total_idx=np.append(total_idx,total_idx);
      np.random.shuffle(total_idx);selected_idx=total_idx[:tr_data_num];
      tr_data[:,i,:]=total_tr_data[selected_idx];

    x_data = tf.placeholder(tf.float32, [None, seq_size, img_size], name="x_data")
    z_prior = tf.placeholder(tf.float32, [None, seq_size, z_size], name="z_prior")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    global_step = tf.Variable(0, name="global_step", trainable=False)

    x_generated, g_params = build_generator(z_prior,keep_prob); 
    y_data, y_generated, d_params = build_discriminator(x_data, x_generated, keep_prob)

    d_loss = - (tf.log(y_data) + tf.log(1 - y_generated))
    g_loss = - tf.log(y_generated)

    optimizer_g = tf.train.AdamOptimizer(0.0001)
    optimizer_d = tf.train.AdamOptimizer(0.00001)
    #optimizer_d = tf.train.AdamOptimizer(0.000001)

    d_trainer = optimizer_d.minimize(d_loss, var_list=d_params)
    g_trainer = optimizer_g.minimize(g_loss, var_list=g_params)

    # tensorboard
    tf.summary.scalar('desc_loss',tf.reduce_sum(d_loss));
    tf.summary.scalar('gen_loss',tf.reduce_sum(g_loss));
    merged=tf.summary.merge_all()
    writer=tf.summary.FileWriter(log_dir)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    sess = tf.Session()

    sess.run(init)

    if to_restore:
        chkpt_fname = tf.train.latest_checkpoint(output_path)
        saver.restore(sess, chkpt_fname)
    else:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)

    z_sample_val = np.random.normal(0, 1, size=(batch_size, seq_size, z_size)).astype(np.float32)
    z_sample_val[:,1:,:]=0.0;

    for i in range(sess.run(global_step), max_epoch):
        for j in range(tr_data_num / batch_size):
            print "epoch:%s, iter:%s" % (i, j)
            x_value = tr_data[j*batch_size:(j+1)*batch_size];
            x_value = 2 * x_value.astype(np.float32) - 1
            z_value = np.random.normal(0, 1, size=(batch_size, seq_size, z_size)).astype(np.float32)
            z_value[:,1:,:]=0.0;
            sess.run(d_trainer,
                     feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
            if j % 1 == 0:
                summary,_=sess.run([merged,g_trainer],
                     feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
                writer.add_summary(summary,(i*(tr_data_num/batch_size)+j))
        for j in range(seq_size):
          x_gen_val = sess.run(x_generated[j], feed_dict={z_prior: z_sample_val,keep_prob:np.sum(0.7).astype(np.float32)})
          show_result(x_gen_val, "output/sample"+str(i)+"_"+str(j)+".jpg")
        sess.run(tf.assign(global_step, i + 1))

def test():
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
    x_generated, _ = build_generator(z_prior)
    chkpt_fname = tf.train.latest_checkpoint(output_path)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(init)
    saver.restore(sess, chkpt_fname)
    z_test_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
    x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_test_value})
    show_result(x_gen_val, "output/test_result.jpg")


if __name__ == '__main__':
    if to_train:
        train()
    else:
        test()
