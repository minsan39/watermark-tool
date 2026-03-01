Set-Location "d:\Python项目"

$gitPath = "C:\Program Files\Git\bin\git.exe"

Write-Host "=== Git Init ===" -ForegroundColor Green
& $gitPath init

Write-Host "`n=== Config User ===" -ForegroundColor Green
& $gitPath config user.email "user@example.com"
& $gitPath config user.name "User"

Write-Host "`n=== Git Status ===" -ForegroundColor Green
& $gitPath status

Write-Host "`n=== Git Add ===" -ForegroundColor Green
& $gitPath add .

Write-Host "`n=== Git Commit ===" -ForegroundColor Green
& $gitPath commit -m "feat: add LaMa deep learning watermark remover"

Write-Host "`n=== Git Log ===" -ForegroundColor Green
& $gitPath log --oneline

Write-Host "`n=== Done! ===" -ForegroundColor Green
