"""Unchanged DWB formula; historical training loss configuration is unconfirmed."""
import numpy as np


def custom_loss_dwb(loss_0_previous, loss_0_current, loss_1_previous, loss_1_current, T=1):
    """ 
    DWB 
    第一、第三个参数是上一代 两个任务(任务0,任务1)的loss, 
    第二、第四个参数是这一代 两个任务(任务0,任务1)的loss,
    输出是下一代两个任务Loss函数的权重
    """
    N = 2 # num of tasks
    r0_previous = loss_0_current / loss_0_previous # loss ratio for task_0
    r1_previous = loss_1_current / loss_1_previous # loss ratio for task_1
    element_0 = np.exp(r0_previous/T)
    element_1 = np.exp(r1_previous/T)
    total = element_0 + element_1
    return N*element_0 / total, N*element_1 / total
