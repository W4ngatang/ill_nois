import subprocess

exp_batch_prefix = 'iclr_noMWU'
n_bin_search_steps = 3
n_opt_steps = 100
use_mwu = 1

mwu_penalties = [.1, .25]
n_mwu_stepss = [10]
models = ['resnet152', 'alex', 'vgg19bn', 'squeeze1_1', 'dense161']
targets = ['none', 'least']
opt_consts = [1.]#, 10., 100., 1000.]
learning_rates = [.001, .01, .1]#, 1.]
stochastic_generate = [0]#, 1]

if use_mwu:
    for lr in learning_rates:
        for const in opt_consts:
            for model in models:
                for target in targets:
                    for n_mwu_steps in n_mwu_stepss:
                        for mwu_penalty in mwu_penalties:
                            exp_name = "%s_holdout_%s_targ_%s_const_%.3f_lr_%.3f_%d_mwu_steps_%.3f_penalty" % (exp_batch_prefix, model, target, const, lr, n_mwu_steps, mwu_penalty)
                            subprocess.call(["sbatch", "src/run_job.sh", 
                                            exp_name, str(n_opt_steps),
                                            str(n_bin_search_steps),
                                            model, target, str(const), str(lr),
                                            str(use_mwu), str(n_mwu_steps), 
                                            str(mwu_penalty)
                                            ])

else:
    n_mwu_steps, mwu_penalty = 0, 0.1
    for lr in learning_rates:
        for const in opt_consts:
            for model in models:
                for target in targets:
                        exp_name = "%s_holdout_%s_targ_%s_const_%.3f_lr_%.3f" % (exp_batch_prefix, model, target, const, lr)
                        subprocess.call(["sbatch", "src/run_job.sh", exp_name,
                                        str(n_opt_steps),
                                        str(n_bin_search_steps),
                                        model, target, str(const), str(lr),
                                        str(use_mwu), str(n_mwu_steps), 
                                        str(mwu_penalty)
                                        ])
