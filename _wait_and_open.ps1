# 等待本地服务就绪后打开浏览器（由 启动系统.bat 调用）
$ErrorActionPreference = 'SilentlyContinue'
$url = 'http://localhost:8501'
for ($i = 0; $i -lt 40; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
        if ($resp.StatusCode -eq 200) {
            Write-Host 'Server is ready, opening browser ...'
            Start-Process $url
            exit 0
        }
    } catch { }
    Start-Sleep -Milliseconds 700
}
Write-Host 'Server did not respond in time. Please open http://localhost:8501 manually.'
exit 1
