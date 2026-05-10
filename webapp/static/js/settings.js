const SettingsManager = (() => {
    const STORAGE_KEY = "readingbuddy_settings";
    const KEY_STORAGE_KEY = "readingbuddy_key";

    async function _deriveKey(password) {
        const enc = new TextEncoder();
        const keyMaterial = await crypto.subtle.importKey(
            "raw", enc.encode(password), "PBKDF2", false, ["deriveKey"]
        );
        const salt = await crypto.subtle.digest(
            "SHA-256", enc.encode("readingbuddy-salt-v1")
        );
        return crypto.subtle.deriveKey(
            { name: "PBKDF2", salt, iterations: 100000, hash: "SHA-256" },
            keyMaterial,
            { name: "AES-GCM", length: 256 },
            false,
            ["encrypt", "decrypt"]
        );
    }

    async function _encrypt(data) {
        const enc = new TextEncoder();
        const key = await _deriveKey(KEY_STORAGE_KEY);
        const iv = crypto.getRandomValues(new Uint8Array(12));
        const ciphertext = await crypto.subtle.encrypt(
            { name: "AES-GCM", iv },
            key,
            enc.encode(JSON.stringify(data))
        );
        const combined = new Uint8Array(iv.length + ciphertext.byteLength);
        combined.set(iv);
        combined.set(new Uint8Array(ciphertext), iv.length);
        return btoa(String.fromCharCode(...combined));
    }

    async function _decrypt(encoded) {
        try {
            const raw = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
            const iv = raw.slice(0, 12);
            const ciphertext = raw.slice(12);
            const key = await _deriveKey(KEY_STORAGE_KEY);
            const decrypted = await crypto.subtle.decrypt(
                { name: "AES-GCM", iv },
                key,
                ciphertext
            );
            return JSON.parse(new TextDecoder().decode(decrypted));
        } catch {
            return null;
        }
    }

    async function save(settings) {
        const encrypted = await _encrypt(settings);
        localStorage.setItem(STORAGE_KEY, encrypted);
    }

    async function load() {
        const encrypted = localStorage.getItem(STORAGE_KEY);
        if (!encrypted) return null;
        return _decrypt(encrypted);
    }

    function clear() {
        localStorage.removeItem(STORAGE_KEY);
    }

    function hasConfigured() {
        return !!localStorage.getItem(STORAGE_KEY);
    }

    return { save, load, clear, hasConfigured };
})();
