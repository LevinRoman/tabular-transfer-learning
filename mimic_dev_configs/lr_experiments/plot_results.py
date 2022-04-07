import matplotlib.pyplot as plt
import json

def plot_setup(file_path, ax, model_name):
    test_auc = []
    train_auc = []
    with open(file_path, "r") as fp:
        data = json.load(fp)
        for epoch in range(200):
            cur_test_auc = data[f"Epoch_{epoch}_metrics"]["test"]["roc_auc"]
            cur_train_auc = data[f"Epoch_{epoch}_metrics"]["train"]["roc_auc"]
            test_auc.append(cur_test_auc)
            train_auc.append(cur_train_auc)

    ax.plot(test_auc, label = model_name+'_test')
    ax.plot(train_auc, label = model_name+'_train')
    return

tab_transformer_default_lr = 'mimic4_tab_transformer_default_lr/stats.json'
tab_transformer = 'mimic4_tab_transformer/stats.json'
resnet = 'mimic4_resnet/stats.json'
resnet_default_lr = 'mimic4_resnet_default_lr/stats.json'
resnet_experiment = '/cmlscratch/vcherepa/rilevin/tabular/tabular-transfer-learning/mimic/tabular-transfer-learning/RTDL/output_mimic_dev/deep_dwnstrm_tuned_standard/mimic4_downstream_100samples_tuned_mlp_head_from_supervised_pretrain_resnet_seed0/stats.json'
if __name__ == '__main__':
    fig, ax = plt.subplots()
    # plot_setup(tab_transformer_default_lr, ax, 'tab_transformer_default_lr')
    # plot_setup(tab_transformer, ax, 'tab_transformer')
    plot_setup(resnet_experiment, ax, 'resnet_Experiment')
    plot_setup(resnet_default_lr, ax, 'resnet_default_lr')
    plot_setup(resnet, ax, 'resnet')
    plt.legend()
    plt.savefig('mimic4_tab_transformer_resnet.png')
