CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -O3 -march=native -ffast-math -funroll-loops -mfma
INCLUDES = -Iinclude -Isrc
SOURCES = src/memory.c src/matrix.c src/tensor.c src/model.c src/attention.c src/transformer.c src/sampler.c src/tokenizer.c src/qwen3_inference.c src/qwen3_inference_internal.c
HEADERS = include/memory.h include/qwen3_inference.h include/matrix.h include/tensor.h include/attention.h include/sampler.h include/transformer.h include/tokenizer.h include/model_internal.h
OBJECTS = $(SOURCES:.c=.o)
TARGET = libqwen3_inference.a

# Test targets
TEST_MODEL_SOURCES = tests/test_model_loading.c
TEST_MODEL_TARGET = test_model_loading
TEST_MATRIX_SOURCES = tests/test_matrix.c
TEST_MATRIX_TARGET = test_matrix
TEST_ATTENTION_SOURCES = tests/test_attention.c
TEST_ATTENTION_TARGET = test_attention
TEST_TRANSFORMER_SOURCES = tests/test_transformer.c
TEST_TRANSFORMER_TARGET = test_transformer
TEST_INFERENCE_SOURCES = tests/test_inference.c
TEST_INFERENCE_TARGET = test_inference
TEST_COMPLETE_SOURCES = tests/test_complete_inference.c
TEST_COMPLETE_TARGET = test_complete_inference

all: $(TARGET) $(TEST_MODEL_TARGET) $(TEST_MATRIX_TARGET) $(TEST_ATTENTION_TARGET) $(TEST_TRANSFORMER_TARGET) $(TEST_INFERENCE_TARGET) $(TEST_COMPLETE_TARGET)

$(TARGET): $(OBJECTS)
	ar rcs $@ $^

$(TEST_MODEL_TARGET): $(TEST_MODEL_SOURCES) $(TARGET)
	$(CC) $(CFLAGS) -o $@ $(TEST_MODEL_SOURCES) $(TARGET) $(INCLUDES) -lpthread

$(TEST_MATRIX_TARGET): $(TEST_MATRIX_SOURCES) $(TARGET)
	$(CC) $(CFLAGS) -o $@ $(TEST_MATRIX_SOURCES) $(TARGET) $(INCLUDES) -lpthread -lm

$(TEST_ATTENTION_TARGET): $(TEST_ATTENTION_SOURCES) $(TARGET)
	$(CC) $(CFLAGS) -o $@ $(TEST_ATTENTION_SOURCES) $(TARGET) $(INCLUDES) -lpthread -lm

$(TEST_TRANSFORMER_TARGET): $(TEST_TRANSFORMER_SOURCES) $(TARGET)
	$(CC) $(CFLAGS) -o $@ $(TEST_TRANSFORMER_SOURCES) $(TARGET) $(INCLUDES) -lpthread -lm

$(TEST_INFERENCE_TARGET): $(TEST_INFERENCE_SOURCES) $(TARGET)
	$(CC) $(CFLAGS) -o $@ $(TEST_INFERENCE_SOURCES) $(TARGET) $(INCLUDES) -lpthread -lm

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

src/memory.o: src/memory.c src/memory.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

src/matrix.o: src/matrix.c include/matrix.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

src/tensor.o: src/tensor.c include/tensor.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

src/model.o: src/model.c include/qwen3_inference.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

src/attention.o: src/attention.c include/attention.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

src/sampler.o: src/sampler.c include/sampler.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

src/tokenizer.o: src/tokenizer.c include/tokenizer.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

src/transformer.o: src/transformer.c include/transformer.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

test-model: $(TEST_MODEL_TARGET)
	./$(TEST_MODEL_TARGET)

test-matrix: $(TEST_MATRIX_TARGET)
	./$(TEST_MATRIX_TARGET)

test: test-model test-matrix test-attention test-transformer test-inference

test-transformer: $(TEST_TRANSFORMER_TARGET)
	./$(TEST_TRANSFORMER_TARGET)

test-inference: $(TEST_INFERENCE_TARGET)
	./$(TEST_INFERENCE_TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET) $(TEST_MODEL_TARGET) $(TEST_MATRIX_TARGET) $(TEST_ATTENTION_TARGET) $(TEST_TRANSFORMER_TARGET) $(TEST_INFERENCE_TARGET)

.PHONY: all clean test

$(TEST_COMPLETE_TARGET): $(TEST_COMPLETE_SOURCES) $(TARGET)
	$(CC) $(CFLAGS) -o $@ $(TEST_COMPLETE_SOURCES) $(TARGET) $(INCLUDES) -lpthread -lm

test-complete: $(TEST_COMPLETE_TARGET)
	./$(TEST_COMPLETE_TARGET)

test-attention: $(TEST_ATTENTION_TARGET)
	./$(TEST_ATTENTION_TARGET)