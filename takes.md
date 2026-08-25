# Notes on my projects:

## Notes on my first vector add:

- Learned how to create a simple kernel to add two vectors on the GPU
- Still learning the cuda syntax and all of the built in functions
- Also created a benchmark timing harness to assess the speed
- Had issue at first where it was too fast (due to arrays being small enough to fit in cache)
- Arithmetic intensity is 1 flop / 12 bytes = .083 flops/byte
