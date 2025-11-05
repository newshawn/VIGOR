# 1) 代理参数（按你现在的 HTTP 代理）
PROXY_URL="http://10.130.130.5:7891"

# 2) 当前会话 + 后续会话统一继承
{
  echo "export HTTP_PROXY=\"$PROXY_URL\""
  echo "export HTTPS_PROXY=\"$PROXY_URL\""
  echo "export http_proxy=\"$PROXY_URL\""
  echo "export https_proxy=\"$PROXY_URL\""
  echo 'export NO_PROXY="127.0.0.1,localhost,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.local"'
} | tee -a ~/.bashrc >/dev/null
source ~/.bashrc

# 3) 写入 VS Code 远端 settings.json（让 Electron/Node 网络栈也走代理）
mkdir -p ~/.vscode-server/data/Machine
SETTINGS=~/.vscode-server/data/Machine/settings.json
if [ -f "$SETTINGS" ]; then
  cp "$SETTINGS" "$SETTINGS.bak.$(date +%s)"
fi
cat > "$SETTINGS" <<EOF
{
  "http.proxy": "$PROXY_URL",
  "http.proxySupport": "on",
  "http.proxyStrictSSL": false,
  "extensions.autoCheckUpdates": false
}
EOF

# 4) 连通性自检（握手通就行；401/JSON 也算通）
curl -I https://chatgpt.com || true
curl -4 -I https://chatgpt.com || true
echo ">>> If both above show HTTP response headers, proxy path is OK."

# 5) 杀掉并重启 VS Code Server（从本地 VS Code 执行命令更好；这里给 CLI 版兜底）
# 在本机 VS Code 命令面板运行：Remote-SSH: Kill VS Code Server on Host <server>，然后重新连接。
echo ">>> 在本机 VS Code：按 F1 -> Remote-SSH: Kill VS Code Server on Host..., 随后重连。"