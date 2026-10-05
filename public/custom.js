/**
 * 前端功能載入器。
 *
 * Chainlit 的 custom_js 只接受單一檔案，但本專案有多項獨立的前端功能，
 * 因此由這支腳本依序載入，各功能維持各自的檔案。
 */
["/public/remove-watermark.js", "/public/download-conversation.js"].forEach(function (src) {
  var script = document.createElement("script");
  script.src = src;
  document.head.appendChild(script);
});
