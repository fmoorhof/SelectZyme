import pytest

from src.logic import Stack


@pytest.fixture(name="stack_inst")
def stack_fixture():
    return Stack()


def test_constructor(stack_inst):
    assert isinstance(stack_inst, Stack)
    assert len(stack_inst) == 0


def test_push(stack_inst):
    stack_inst.push(3)
    assert len(stack_inst) == 1
    stack_inst.push(5)
    assert len(stack_inst) == 2


def test_pop(stack_inst):
    stack_inst.push("abc")
    stack_inst.push("def")
    assert stack_inst.pop() == "def"
    assert len(stack_inst) == 1
    assert stack_inst.pop() == "abc"
    assert len(stack_inst) == 0
    with pytest.raises(IndexError, match=r".*stack is empty.*"):
        stack_inst.pop()