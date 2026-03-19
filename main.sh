#!/bin/bash

CONTAINER_NAME="conversion_v3"
DOCKER_COMPOSE_FILE="docker/docker-compose.conversion.yml"
PROJECT_NAME="conversion_v3"

# BuildKit 활성화 (더 빠른 캐시 활용)
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# 호스트 UID/GID 전달 → 컨테이너 내 data 폴더가 호스트 유저 소유로 생성됨
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

echo "── V3 Conversion ──"
echo ""

while true; do
  # 컨테이너 실행 여부 체크
  if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "1) Cache build & run"
    echo "2) 🟢 Connect to container"
  else
    echo "1) Cache build & run"
    echo "2) 🔴 Connect to container"
  fi

  echo "3) Docker Force Kill"
  echo "4) List docker containers"
  echo "5) No-cache rebuild & run"
  echo "6) Exit (or press ESC)"
  echo -n "Choice: "

  read -rsn1 c

  if [[ $c == $'\e' ]]; then
    echo -e "\n[ESC] Exit"
    break
  fi

  echo ""

  if [[ "$c" == "1" || -z "$c" ]]; then
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
      echo "⚠️  Container '${CONTAINER_NAME}' is already running."
      read -p "Do you want to restart it? (y/N): " yn
      if [[ "$yn" == "y" || "$yn" == "Y" ]]; then
        echo "🔄 Restarting container..."
        docker compose -p "${PROJECT_NAME}" -f "${DOCKER_COMPOSE_FILE}" up --build -d
      else
        echo "➡️  Keeping the existing container running."
      fi
    else
      docker compose -p "${PROJECT_NAME}" -f "${DOCKER_COMPOSE_FILE}" up --build -d
    fi

  elif [[ "$c" == "2" ]]; then
    docker exec -it "${CONTAINER_NAME}" bash
  elif [[ "$c" == "3" ]]; then
    docker compose -p "${PROJECT_NAME}" -f "${DOCKER_COMPOSE_FILE}" down
  elif [[ "$c" == "4" ]]; then
    docker ps

  elif [[ "$c" == "5" ]]; then

    docker compose -p "${PROJECT_NAME}" -f "${DOCKER_COMPOSE_FILE}" build --no-cache
    docker compose -p "${PROJECT_NAME}" -f "${DOCKER_COMPOSE_FILE}" up --remove-orphans -d

  elif [[ "$c" == "6" ]]; then
    echo "Exit"
    break

  else
    echo "Invalid choice"
  fi

  echo ""

done
