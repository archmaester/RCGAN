import numpy as np
import tensorflow as tf
import pdb
import random
import json
from scipy.stats import mode

import data_utils
import plotting
import model
import utils

from time import time
from math import floor
from mmd import rbf_mmd2, median_pairwise_distance, mix_rbf_mmd2_and_ratio

tf.logging.set_verbosity(tf.logging.ERROR)
with tf.device('/gpu:0'):
	identifier = 'mnistfull'
	settings = utils.load_settings_from_file(identifier)

	samples, pdf, labels = data_utils.get_samples_and_labels(settings)

	locals().update(settings)
	# json.dump(settings, open('./experiments/settings/' + identifier + '.txt', 'w'), indent=0)

	data_path = './experiments/data/' + identifier + '.data.npy'
	np.save(data_path, {'samples': samples, 'pdf': pdf, 'labels': labels})
	print('Saved training data to', data_path)


	# --- build model --- #

	Z, X, CG, CD, CS = model.create_placeholders(batch_size, seq_length, latent_dim, 
										num_signals, cond_dim)

	discriminator_vars = ['hidden_units_d', 'seq_length', 'cond_dim', 'batch_size', 'batch_mean']
	discriminator_settings = dict((k, settings[k]) for k in discriminator_vars)
	generator_vars = ['hidden_units_g', 'seq_length', 'batch_size', 
					  'num_generated_features', 'cond_dim', 'learn_scale']
	generator_settings = dict((k, settings[k]) for k in generator_vars)

	CGAN = (cond_dim > 0)
	print(CGAN)
	D_loss, G_loss, accuracy = model.GAN_loss(Z, X, generator_settings, discriminator_settings, CGAN, CG, CD, CS, wrong_labels=wrong_labels)
	D_solver, G_solver = model.GAN_solvers(D_loss, G_loss, learning_rate, batch_size, total_examples=samples['train'].shape[0], l2norm_bound=0, batches_per_lot=0, sigma=0, dp=False)
	G_sample = model.generator(Z, **generator_settings, reuse=True, c=CG)

	print("Train Samples : ", samples['train'].shape)
	print("Validation Samples : ", samples['vali'].shape)
	print("Test Samples : ", samples['test'].shape)
	print("Train Labels :", labels['train'])
	# # --- evaluation --- #

	# # # frequency to do visualisations
	vis_freq = 10
	eval_freq = 1

	# # # get heuristic bandwidth for mmd kernel from evaluation samples
	heuristic_sigma_training = median_pairwise_distance(samples['vali'])
	best_mmd2_so_far = 1000

	# # optimise sigma using that (that's t-hat)
	n_samples_sigma = len(samples['vali'])
	batch_multiplier = n_samples_sigma//batch_size
	eval_size = batch_multiplier*batch_size
	print(eval_size)
	eval_eval_size = int(0.2*eval_size)
	eval_real_PH = tf.placeholder(tf.float32, [eval_eval_size, seq_length, num_generated_features])
	eval_sample_PH = tf.placeholder(tf.float32, [eval_eval_size, seq_length, num_generated_features])
	n_sigmas = 2
	sigma = tf.get_variable(name='sigma', shape=n_sigmas, initializer=tf.constant_initializer(value=np.power(heuristic_sigma_training, np.linspace(-1, 3, num=n_sigmas))))
	mmd2, that = mix_rbf_mmd2_and_ratio(eval_real_PH, eval_sample_PH, sigma)
	with tf.variable_scope("SIGMA_optimizer"):
		sigma_solver = tf.train.RMSPropOptimizer(learning_rate=0.05).minimize(-that, var_list=[sigma])
		#sigma_solver = tf.train.AdamOptimizer().minimize(-that, var_list=[sigma])
		#sigma_solver = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(-that, var_list=[sigma])
	sigma_opt_iter = 2000
	sigma_opt_thresh = 0.001
	sigma_opt_vars = [var for var in tf.global_variables() if 'SIGMA_optimizer' in var.name]

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
sess.run(tf.global_variables_initializer())

vis_Z = model.sample_Z(batch_size, seq_length, latent_dim, use_time)
vis_C = model.sample_C(batch_size, cond_dim, max_val, one_hot)
# vis_C[:1] = np.arange(cond_dim)
vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z, CG: vis_C})

vis_real_indices = np.random.choice(len(samples['vali']), size=6)
vis_real = np.float32(samples['vali'][vis_real_indices, :, :])
vis_real_labels = labels['vali'][vis_real_indices]

samps = vis_real
labs = vis_real_labels 

plotting.save_mnist_plot_sample(samps.reshape(-1, seq_length**2, 1), 0, identifier + '_real', n_samples=6, labels=labs)

trace = open('./experiments/traces/' + identifier + '.trace.txt', 'w')
trace.write('epoch time D_loss G_loss mmd2\n')

# # --- train --- #
train_vars = ['batch_size', 'D_rounds', 'G_rounds', 'use_time', 'seq_length', 
              'latent_dim', 'num_generated_features', 'cond_dim', 'max_val', 
              'WGAN_clip', 'one_hot']
train_settings = dict((k, settings[k]) for k in train_vars)


t0 = time()
best_epoch = 0
print('epoch\ttime\tD_loss\tG_loss\tmmd2\t')

for epoch in range(num_epochs):
    D_loss_curr, G_loss_curr = model.train_epoch(epoch, samples['train'], labels['train'],
                                        sess, Z, X, CG, CD, CS,accuracy,
                                        D_loss, G_loss,
                                        D_solver, G_solver, 
                                        **train_settings)
    # -- eval -- #
    # visualise plots of generated samples, with/without labels
    if epoch % vis_freq == 0:
        if CGAN:
            vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z, CG: vis_C})
        else:
            vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z})
        plotting.visualise_at_epoch(vis_sample, data, 
                predict_labels, one_hot, epoch, identifier, num_epochs,
                resample_rate_in_min, multivariate_mnist, seq_length, labels=vis_C)

    # compute mmd2 and, if available, prob density
    if epoch % eval_freq == 0:
        ## how many samples to evaluate with?
        eval_Z = model.sample_Z(eval_size, seq_length, latent_dim, use_time)
        eval_C = model.sample_C(eval_size, cond_dim, max_val, one_hot)
        eval_sample = np.empty(shape=(eval_size, seq_length, num_signals))

        for i in range(batch_multiplier):
            if CGAN:
                eval_sample[i*batch_size:(i+1)*batch_size, :, :] = sess.run(G_sample, feed_dict={Z: eval_Z[i*batch_size:(i+1)*batch_size], CG: eval_C[i*batch_size:(i+1)*batch_size]})
            else:
                eval_sample[i*batch_size:(i+1)*batch_size, :, :] = sess.run(G_sample, feed_dict={Z: eval_Z[i*batch_size:(i+1)*batch_size]})
        eval_sample = np.float32(eval_sample)
        eval_real = np.float32(samples['vali'][np.random.choice(len(samples['vali']), size=batch_multiplier*batch_size), :, :])

        eval_eval_real = eval_real[:eval_eval_size]
        eval_test_real = eval_real[eval_eval_size:]
        eval_eval_sample = eval_sample[:eval_eval_size]
        eval_test_sample = eval_sample[eval_eval_size:]

        ## MMD
        # reset ADAM variables
        sess.run(tf.initialize_variables(sigma_opt_vars))
        sigma_iter = 0
        that_change = sigma_opt_thresh*2
        old_that = 0
        while that_change > sigma_opt_thresh and sigma_iter < sigma_opt_iter:
            new_sigma, that_np, _ = sess.run([sigma, that, sigma_solver], feed_dict={eval_real_PH: eval_eval_real, eval_sample_PH: eval_eval_sample})
#             print(new_sigma,that_np)
            that_change = np.abs(that_np - old_that)
            old_that = that_np
            sigma_iter += 1
        opt_sigma = sess.run(sigma)
        mmd2, that_np = sess.run(mix_rbf_mmd2_and_ratio(eval_test_real, eval_test_sample,biased=False, sigmas=sigma))
        
        ## save parameters
        if mmd2 < best_mmd2_so_far and epoch > 10:
            best_epoch = epoch
            best_mmd2_so_far = mmd2
            model.dump_parameters(identifier + '_' + str(epoch), sess)

        pdf_sample = 'NA'
        pdf_real = 'NA'
    else:
        # report nothing this epoch
        mmd2 = 'NA'
        that = 'NA'
        pdf_sample = 'NA'
        pdf_real = 'NA'

    t = time() - t0
    try:
        print('%d\t%.2f\t%.4f\t%.4f\t%.5f\t' % (epoch, t, D_loss_curr, G_loss_curr, mmd2))
    except TypeError:       # pdf are missing (format as strings)
        print('%d\t%.2f\t%.4f\t%.4f\t%s' % (epoch, t, D_loss_curr, G_loss_curr, mmd2))

    ## save trace
#     trace.write(' '.join(map(str, [epoch, t, D_loss_curr, G_loss_curr])) + '\n')
#     if epoch % 10 == 0: 
#         trace.flush()
#         plotting.plot_trace(identifier, xmax=num_epochs, dp=False)

    if shuffle:     # shuffle the training data 
        perm = np.random.permutation(samples['train'].shape[0])
        samples['train'] = samples['train'][perm]
        if labels['train'] is not None:
            labels['train'] = labels['train'][perm]

    if epoch % 50 == 0:
        model.dump_parameters(identifier + '_' + str(epoch), sess)
            
trace.flush()
plotting.plot_trace(identifier, xmax=num_epochs, dp=False)
model.dump_parameters(identifier + '_' + str(epoch), sess)

# # ## after-the-fact evaluation
# # #n_test = vali.shape[0]      # using validation set for now TODO
# # #n_batches_for_test = floor(n_test/batch_size)
# # #n_test_eval = n_batches_for_test*batch_size
# # #test_sample = np.empty(shape=(n_test_eval, seq_length, num_signals))
# # #test_Z = model.sample_Z(n_test_eval, seq_length, latent_dim, use_time)
# # #for i in range(n_batches_for_test):
# # #    test_sample[i*batch_size:(i+1)*batch_size, :, :] = sess.run(G_sample, feed_dict={Z: test_Z[i*batch_size:(i+1)*batch_size]})
# # #test_sample = np.float32(test_sample)
# # #test_real = np.float32(vali[np.random.choice(n_test, n_test_eval, replace=False), :, :])
# # ## we can only get samples in the size of the batch...
# # #heuristic_sigma = median_pairwise_distance(test_real, test_sample)
# # #test_mmd2, that = sess.run(mix_rbf_mmd2_and_ratio(test_real, test_sample, sigmas=heuristic_sigma, biased=False))
# # ##print(test_mmd2, that)
