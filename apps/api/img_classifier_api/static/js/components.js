/**
 * Shared Alpine.js component factories used across all pages.
 *
 * Usage in templates:
 *   x-data="fileUpload()"       — drag-and-drop image upload zone
 *   x-data="modelSelector()"    — model list / activation
 *   x-data="alertBanner()"      — inline dismissible alert
 */

/**
 * Lightweight file-upload zone.
 *
 * Exposes:
 *   selectedFile, previewUrl, isDragging
 *   handleFileSelect(event), handleDrop(event), clearFile()
 */
function fileUpload() {
    return {
        selectedFile: null,
        previewUrl: null,
        isDragging: false,

        handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) this._setFile(file);
        },

        handleDrop(event) {
            this.isDragging = false;
            const file = event.dataTransfer.files[0];
            if (file && file.type.startsWith("image/")) this._setFile(file);
        },

        _setFile(file) {
            if (this.previewUrl) URL.revokeObjectURL(this.previewUrl);
            this.selectedFile = file;
            this.previewUrl = URL.createObjectURL(file);
        },

        clearFile() {
            if (this.previewUrl) URL.revokeObjectURL(this.previewUrl);
            this.selectedFile = null;
            this.previewUrl = null;
        },
    };
}

/**
 * Model list + activation selector.
 *
 * Exposes:
 *   models[], currentModel, loading
 *   loadModels(), activateModel(name)
 */
function modelSelector() {
    return {
        models: [],
        currentModel: null,
        loading: false,

        async init() {
            await this.loadModels();
            window.addEventListener("ws:model_loaded", () => this.loadModels());
        },

        async loadModels() {
            this.loading = true;
            try {
                const data = await window.utils.api.get("/api/models");
                this.models = data.models || [];
                this.currentModel = data.current_model;
            } catch (e) {
                console.error("modelSelector: failed to load models", e);
            } finally {
                this.loading = false;
            }
        },

        async activateModel(name) {
            if (name === this.currentModel) return;
            try {
                await window.utils.api.post(`/api/models/${name}/activate`);
                this.currentModel = name;
                if (window.Alpine) Alpine.store("app").setActiveModel(name);
                window.showToast(`Switched to ${name}`, "success");
                document.body.dispatchEvent(new CustomEvent("modelChange", { detail: { name } }));
            } catch (e) {
                window.showToast("Failed to switch model", "error");
            }
        },
    };
}

/**
 * Simple inline dismissible alert banner.
 *
 * Exposes:
 *   visible, message, type   (type: "success" | "error" | "warning" | "info")
 *   show(message, type), dismiss()
 */
function alertBanner() {
    return {
        visible: false,
        message: "",
        type: "info",

        show(message, type = "info") {
            this.message = message;
            this.type = type;
            this.visible = true;
        },

        dismiss() {
            this.visible = false;
        },
    };
}

// Register as globals so templates can reference them by name
if (typeof window !== "undefined") {
    window.fileUpload = fileUpload;
    window.modelSelector = modelSelector;
    window.alertBanner = alertBanner;
}
