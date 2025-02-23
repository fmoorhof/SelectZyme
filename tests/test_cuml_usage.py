from io import StringIO

import pytest
import torch
import cudf, requests


@pytest.mark.tryfirst  # dont stop testing if this fails
def test_torch_cuda_availability():
    """
    This test asserts if PyTorch can detect CUDA-enabled GPUs.
    """
    if torch.cuda.is_available():
        print(f"GPUs are available. Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        assert torch.cuda.device_count() > 0
    else:
        print("GPUs are not available.")
        assert False, "CUDA-enabled GPUs are not available."


def test_cuml_usage():
    """
    This test asserts if RAPIDSAI CuML can be used for GPU accelerated programming.
    """
    url = "https://github.com/plotly/datasets/raw/master/tips.csv"
    content = requests.get(url, timeout=10).content.decode("utf-8")

    tips_df = cudf.read_csv(StringIO(content))
    tips_df["tip_percentage"] = tips_df["tip"] / tips_df["total_bill"] * 100

    # display average tip by dining party size
    print(tips_df.groupby("size").tip_percentage.mean())

    assert content != ""
