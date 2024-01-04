from python import Python
import math

fn main() raises:
    let torch = Python.import_module("torch")
    let nn = torch.nn
    let F = nn.functional

