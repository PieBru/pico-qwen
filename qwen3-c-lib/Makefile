CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -O3 -march=native -ffast-math -funroll-loops
INCLUDES = -Iinclude -Isrc
SOURCES = src/memory.c
HEADERS = include/memory.h include/qwen3_inference.h
OBJECTS = $(SOURCES:.c=.o)
TARGET = libqwen3_inference.a

# Memory test
TEST_SOURCES = test_memory_standalone.c
TEST_TARGET = test_memory_standalone
TEST_OBJECTS = $(TEST_SOURCES:.c=.o)

all: $(TARGET) $(TEST_TARGET)

$(TARGET): $(OBJECTS)
	ar rcs $@ $^

$(TEST_TARGET): $(TEST_OBJECTS) $(TARGET)
	$(CC) $(CFLAGS) -o $@ $(TEST_OBJECTS) $(TARGET) $(INCLUDES) -lpthread

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

src/memory.o: src/memory.c src/memory.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

src/qwen3_inference.o: src/qwen3_inference.c src/qwen3_inference.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

test_memory_standalone.o: test_memory_standalone.c src/memory.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

test: $(TEST_TARGET)
	./$(TEST_TARGET)

clean:
	rm -f $(OBJECTS) $(TEST_OBJECTS) $(TARGET) $(TEST_TARGET)

.PHONY: all clean test