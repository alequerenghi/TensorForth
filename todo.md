# Project

## Tensor
- Tensor creation from file
- Tensor creation from memory?

## Stack
- enum for command, tensor and string

- stack which uses union to store item
- stack 'consume' function with reference counter

## Main
- read file from disk and execute commands:
    - if [ start parsing tensor
    - if ] stop parsing tensor
    - if 'c' character execute command
    - if '"' parse a string and load tensor from memory
