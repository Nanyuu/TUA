import numpy as np



def Test_attack_success_rate_for_Class_Node(label_OneHot: np.ndarray, output: np.ndarray, attack_node_idx: int):
    """
    Test the ASR on a group of victim nodes
    :param label_OneHot: One hot label
    :param output: output of Nodes
    :param attack_node_idx: attack node Index
    :return:    ASR
    """
    total_victim_num = output.shape[0]  # total victim node number
    success_att_num = 0
    target_label = np.where(label_OneHot[attack_node_idx])[0][0]

    for ii in range(total_victim_num):
        current_label = output[ii].argmax().item()
        if current_label == target_label:
            success_att_num = success_att_num + 1
        else:
            continue

    attack_success_rate = float(success_att_num) / float(total_victim_num)

    return attack_success_rate

