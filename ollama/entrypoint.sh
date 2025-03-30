#!/bin/bash

ollama serve &

sleep 2

# ollama run llama3.2:3b-instruct-fp16

ollama run llama3.2:1b-instruct-fp16

wait