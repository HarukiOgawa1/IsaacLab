#!/bin/bash

# 実行コマンドの基本部分
BASE_COMMAND="./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-G1-v0 --headless"

# 変更したいパラメータと値のリスト
# (例: env.rewards.lin_vel_z_l2.weight の値を変更)
PARAM_NAME="env.rewards.lin_vel_z_l2.weight"
VALUES=(-0.1 -0.2 -0.3)

echo "Isaac Labの学習を異なるパラメータで実行します。"

for val in "${VALUES[@]}"
do
    FULL_COMMAND="${BASE_COMMAND} ${PARAM_NAME}=${val}"
    echo "--- 実行コマンド: ${FULL_COMMAND} ---"
    
    # コマンドを実行
    $FULL_COMMAND
    
    # 実行が成功したか確認（任意）
    if [ $? -eq 0 ]; then
        echo "実行が正常に完了しました。（${PARAM_NAME}=${val}）"
    else
        echo "実行中にエラーが発生しました。（${PARAM_NAME}=${val}）"
    fi
    echo "" # 区切りとして空行を出力
done

echo "全てのパラメータでの実行が完了しました。"