#!/usr/bin/env bash
# MarineGym / Isaac Sim 容器验收脚本
# 新镜像做好后，在容器里跑: bash scripts/preflight_vulkan.sh
# 全部 [PASS] 才说明 Vulkan/RTX 通了，Isaac 能起来。
set -uo pipefail

pass() { echo "[PASS] $1"; }
fail() { echo "[FAIL] $1"; FAILED=1; }
FAILED=0

echo "================ 1. GPU / 驱动注入 ================"
echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-<unset>}  (需含 graphics)"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}          (应为 all 或 GPU UUID，不应是 void)"
case ",${NVIDIA_DRIVER_CAPABILITIES:-}," in
  *,all,*|*,graphics,*) pass "graphics capability 已开启" ;;
  *) fail "缺 graphics capability —— RTX 渲染起不来" ;;
esac
nvidia-smi >/dev/null 2>&1 && pass "nvidia-smi 可用" || fail "nvidia-smi 不可用"

echo "================ 2. 设备节点 ================"
ls /dev/nvidia0 >/dev/null 2>&1 && pass "/dev/nvidia* 存在" || fail "缺 /dev/nvidia*"
ls /dev/dri/renderD* >/dev/null 2>&1 && pass "/dev/dri/renderD* 存在" || fail "缺 DRI render 节点"

echo "================ 3. Vulkan 用户态 ================"
ldconfig -p 2>/dev/null | grep -q "libvulkan.so.1" && pass "Vulkan loader 存在" || fail "缺 libvulkan.so.1"
find /etc/vulkan/icd.d /usr/share/vulkan/icd.d -name 'nvidia_icd*.json' 2>/dev/null | grep -q . \
  && pass "NVIDIA Vulkan ICD 存在" || fail "缺 nvidia_icd.json"

echo "================ 4. vulkaninfo 能否枚举到 L40 (核心) ================"
if command -v vulkaninfo >/dev/null 2>&1; then
  if vulkaninfo --summary 2>/dev/null | grep -iE "deviceName|L40" | grep -qi "L40"; then
    pass "vulkaninfo 枚举到 L40 GPU"
    vulkaninfo --summary 2>/dev/null | grep -iE "deviceName|driverName|driverInfo" | sed 's/^/       /'
  else
    fail "vulkaninfo 未枚举到 L40 —— Vulkan 看不到 GPU"
  fi
else
  fail "vulkaninfo 未安装 (请在镜像里加 vulkan-tools)"
fi

echo "================ 5. Isaac Sim headless 真启动 (决定性) ================"
timeout 180 python - <<'PY'
import sys
try:
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    app.close()
    print("[PASS] SimulationApp(headless) 启动并关闭成功")
    sys.exit(0)
except Exception as e:
    print(f"[FAIL] SimulationApp 报错: {e}")
    sys.exit(1)
PY
RC=$?
if [ $RC -eq 124 ]; then fail "SimulationApp 180s 内挂死 (Vulkan/RTX 初始化卡住 —— 就是当前的病)"
elif [ $RC -ne 0 ]; then fail "SimulationApp 启动失败 (见上方报错)"
fi

echo "=================================================="
[ "${FAILED:-0}" -eq 0 ] && echo "✅ 全部通过，可以跑 train.py" || echo "❌ 有 FAIL，把本输出发给 Niall"
exit "${FAILED:-0}"
