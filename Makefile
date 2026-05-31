TEST_CPP_DIR := ./test/cpp
TEST_CPP_FILES := $(wildcard $(TEST_CPP_DIR)/*.cu)
TEST_CPP_EXES := $(TEST_CPP_FILES:.cu=)

SRC_CPP_DIR := ./src/cpp
SRC_CPP_FILES := $(wildcard $(SRC_CPP_DIR)/*.cu)
SRC_CPP_EXES := $(SRC_CPP_FILES:.cu=)

HEADER_DIR := ./src/cpp/headers

$(TEST_CPP_EXES): %: %.cu
	nvcc -o $@ $< -I $(HEADER_DIR)

$(SRC_CPP_EXES): %: %.cu
	nvcc -o $@ $< -I $(HEADER_DIR)

test-cpp: $(TEST_CPP_EXES)	
	@set -e; for t in $(TESTS); do ./$$t; done
	@echo "All cpp tests passed"

test-py:
	uv run -m pytest

install: 
	uv pip install -e ./src/py/ --no-build-isolation

inference: install
	uv run python ./src/py/model_inference.py

clean:
	rm -rf $(TEST_OBJS)


