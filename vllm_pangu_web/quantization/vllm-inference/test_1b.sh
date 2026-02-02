MASTER_NODE_IP=127.0.0.1  # server node ip

PORT=1040

SERVED_MODEL_NAME=pangu_embedded_1b


curl http://${MASTER_NODE_IP}:${PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'$SERVED_MODEL_NAME'",
        "messages": [
            {
                "role": "user",
                "content": "How to keep healthy. Response in Chinese!"
            }
        ],
        "max_tokens": 512,
        "temperature": 0
    }'
