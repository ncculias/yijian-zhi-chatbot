/**
 * 在標題列加一顆「下載對話」按鈕，把當前對話匯出成 Markdown。
 *
 * 系統目前不保存對話紀錄（見 issue #4），關掉分頁內容就沒了。此功能讓使用者
 * 把問答帶走，貼進研究筆記或報告。
 *
 * 做法刻意全在前端：不碰 Python 的回答流程，萬一失效也只是按鈕沒作用。
 *
 * 注意：本檔依賴 Chainlit 的 DOM 結構（div[id^="step-"]、.ai-message、
 * .message-content）。Chainlit 改版時可能需要調整這幾個選擇器。
 */
(function () {
  "use strict";

  var BUTTON_ID = "download-conversation";
  var FOLLOWUP_HEADING = "你可以接著問";
  var STEP_SELECTOR = 'div[id^="step-"]';

  function pad(n) {
    return String(n).padStart(2, "0");
  }

  function timestamp(sep) {
    var d = new Date();
    return (
      d.getFullYear() + "-" + pad(d.getMonth() + 1) + "-" + pad(d.getDate()) +
      sep + pad(d.getHours()) + (sep === " " ? ":" : "") + pad(d.getMinutes())
    );
  }

  /**
   * 移除本文結尾的「你可以接著問：」標題。
   * 延伸問題本身是獨立的按鈕元素，不在 .message-content 內，不必處理。
   * 只在該字串確實位於結尾時才移除，避免誤刪正文中提到的同樣字句。
   */
  function stripFollowupHeading(text) {
    var i = text.lastIndexOf(FOLLOWUP_HEADING);
    if (i === -1) return text;
    var tail = text.slice(i);
    return tail.length <= FOLLOWUP_HEADING.length + 3 ? text.slice(0, i).trim() : text;
  }

  function collectTurns() {
    var turns = [];
    document.querySelectorAll(STEP_SELECTOR).forEach(function (step) {
      var content = step.querySelector(".message-content");
      var text = (content ? content.innerText : step.innerText).trim();
      if (!text) return;
      var isAssistant = !!step.querySelector(".ai-message");
      text = isAssistant ? stripFollowupHeading(text) : text;
      if (text) turns.push({ role: isAssistant ? "答" : "問", text: text });
    });
    return turns;
  }

  function buildMarkdown(turns) {
    var lines = [
      "# 夷問對話紀錄",
      "",
      "- 日期：" + timestamp(" "),
      "- 網址：" + window.location.origin,
      "",
      "---",
      "",
    ];
    turns.forEach(function (turn) {
      lines.push("## " + turn.role, turn.text, "");
    });
    return lines.join("\n");
  }

  function saveFile(markdown) {
    var blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var link = document.createElement("a");
    link.href = url;
    link.download = "夷問對話_" + timestamp("_") + ".md";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function onClick() {
    try {
      var turns = collectTurns();
      if (!turns.length) {
        console.warn("[下載對話] 目前沒有對話內容");
        return;
      }
      saveFile(buildMarkdown(turns));
    } catch (err) {
      console.error("[下載對話] 匯出失敗：", err);
    }
  }

  /** 沒有對話時停用按鈕，避免下載到空檔案。選擇器很輕量，可在每次 DOM 變動時呼叫。 */
  function refreshState() {
    var button = document.getElementById(BUTTON_ID);
    if (button) button.disabled = !document.querySelector(STEP_SELECTOR);
  }

  function insertButton() {
    if (document.getElementById(BUTTON_ID)) return;
    // 插在主題切換鍵之前；沿用相鄰按鈕的 class，外觀才與 Chainlit 一致
    var anchor = document.getElementById("theme-toggle") || document.getElementById("readme-button");
    if (!anchor || !anchor.parentElement) return;

    var button = document.createElement("button");
    button.id = BUTTON_ID;
    button.type = "button";
    button.className = anchor.className;
    button.textContent = "下載";
    button.title = "把這次對話下載成 Markdown 檔";
    button.addEventListener("click", onClick);
    anchor.parentElement.insertBefore(button, anchor);
  }

  function tick() {
    try {
      insertButton();
      refreshState();
    } catch (err) {
      console.error("[下載對話] 初始化失敗：", err);
    }
  }

  // Chainlit 是單頁應用，標題列可能在腳本執行後才出現，開新對話時也會重建，
  // 因此持續觀察而不中斷 observer。
  new MutationObserver(tick).observe(document.body, { childList: true, subtree: true });
  tick();
})();
