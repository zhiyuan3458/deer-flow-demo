# 🦌 DeerFlow 服务管理

## 快速命令

| 命令 | 用途 |
|------|------|
| `./bootstrap.sh -d` | 启动开发服务（前端 3000，后端 8089） |
| `./stop.sh` | 停止所有 DeerFlow 服务 |
| `./restart.sh` | 重启服务（停止 + 启动） |

## 详细说明

### 启动服务

```bash
# 开发模式（带热重载）
./bootstrap.sh -d

# 生产模式
./bootstrap.sh
```

**启动后：**
- 后端 API: http://localhost:8089
- 前端界面: http://localhost:3000

### 停止服务

```bash
./stop.sh
```

**自动清理：**
- ✅ 停止端口 8089（后端）
- ✅ 停止端口 3000/3001（前端）
- ✅ 清理相关进程（server.py, next dev）

### 重启服务

```bash
./restart.sh
```

等价于：
```bash
./stop.sh
sleep 2
./bootstrap.sh -d
```

## 端口配置

| 服务 | 默认端口 | 配置文件 |
|------|---------|---------|
| 后端 API | 8089 | `server.py` (--port) |
| 前端 | 3000 | Next.js 默认 |
| API URL | - | `.env` (NEXT_PUBLIC_API_URL) |

## 常见问题

### Q: 端口被占用怎么办？

```bash
# 1. 停止服务
./stop.sh

# 2. 检查端口
lsof -i :3000 -i :8089

# 3. 重新启动
./bootstrap.sh -d
```

### Q: 修改配置后需要重启吗？

- ✅ 修改 `.env` → 需要重启
- ✅ 修改 `conf.yaml` → 需要重启
- ✅ 修改代码 → 自动热重载（dev 模式）

### Q: 如何切换端口？

修改以下文件：
1. `server.py` - 第 64 行 `default=8089`
2. `.env` - `NEXT_PUBLIC_API_URL`
3. `web/src/core/api/resolve-service-url.ts` - 第 16、20 行
