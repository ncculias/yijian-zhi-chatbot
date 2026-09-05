# 變更紀錄(Changelog)

本專案所有重要變更都記錄在此檔案。

格式依循 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.1.0/),
版本號依循 [語意化版本](https://semver.org/lang/zh-TW/):
修 bug 進第三位(1.0.x)、新增功能進第二位(1.x.0)、重大改版進第一位(x.0.0)。

分類標籤:`Added` 新增、`Changed` 變更、`Fixed` 修正、`Removed` 移除。

## [未發布]

<!-- 已合併但尚未部署上線的變更寫在這裡,部署時移到新版本號下 -->

## [1.0.1] - 2026-09-05

### Fixed

- 輸入框「夷間志」誤植修正為「夷堅志」(#3, PR #9)
- `.gitignore` 翻譯檔白名單寫法修正,使 zh-TW.json 可正常追蹤(PR #9)

### Added

- 工程管理文件:開發流程、PRD/ADR 範本、迴歸測試清單、本 CHANGELOG(PR #8)

## [1.0.0] - 2026-07-27

### Added

- 首次雲端部署上線:AWS Lightsail(東京)+ Cloudflare(https://yijianzhi.org、自動 HTTPS)
- 程式碼零改動部署;伺服器端僅新增 `.env` 與 `docker-compose.override.yml`(port 80 對應)
- 架構決策紀錄見 [docs/adr/0001-部署架構選擇.md](docs/adr/0001-部署架構選擇.md)
