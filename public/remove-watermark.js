function initWatermarkHiding() {
    function overrideTargetElement() {
    const targetElement = document.querySelector('.MuiStack-root.watermark') || document.querySelector('.watermark');
    console.log(targetElement);
if (targetElement) {
    targetElement.style.display = 'none';

    console.log('Watermark hidden successfully!');
} else {
    console.error('Target element not found!');
}
}

const observer = new MutationObserver(function (mutationsList, observer) {
for (const mutation of mutationsList) {
    if (mutation.type === 'childList') {
        const targetElement = document.querySelector('.MuiStack-root.watermark') || document.querySelector('.watermark');
        if (targetElement) {
            overrideTargetElement();
            observer.disconnect();
        }
    }
}
});

observer.observe(document.body, { childList: true, subtree: true });
    overrideTargetElement();
}

// 本檔可能在 DOMContentLoaded 之後才由 custom.js 動態載入，
// 屆時該事件不會再觸發，需直接初始化。
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initWatermarkHiding);
} else {
    initWatermarkHiding();
}
