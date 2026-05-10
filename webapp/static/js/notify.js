const Notify = (() => {
    const container = document.getElementById("notify-container");

    function create(type, message, duration = 4000) {
        const el = document.createElement("div");
        el.className = `notify-toast notify-${type}`;
        el.innerHTML = `
            <span class="notify-icon">${getIcon(type)}</span>
            <span class="notify-message">${message}</span>
            <button class="notify-close" onclick="this.parentElement.remove()">&times;</button>
        `;
        container.appendChild(el);
        requestAnimationFrame(() => el.classList.add("notify-show"));
        if (duration > 0) {
            setTimeout(() => dismiss(el), duration);
        }
        return el;
    }

    function getIcon(type) {
        const icons = {
            success: "&#10003;",
            error: "&#10007;",
            warning: "&#9888;",
            info: "&#8505;",
        };
        return icons[type] || icons.info;
    }

    function dismiss(el) {
        el.classList.remove("notify-show");
        el.classList.add("notify-hide");
        setTimeout(() => el.remove(), 300);
    }

    return {
        success(msg, dur) { create("success", msg, dur); },
        error(msg, dur) { create("error", msg, dur || 6000); },
        warning(msg, dur) { create("warning", msg, dur); },
        info(msg, dur) { create("info", msg, dur); },
    };
})();
