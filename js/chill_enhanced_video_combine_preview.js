import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "ChillEnhancedVideoCombine";

function assetUrl(asset) {
    const params = new URLSearchParams({
        filename: asset.filename,
        subfolder: asset.subfolder || "",
        type: asset.type || "output",
    });
    return api.apiURL(`/view?${params}`);
}

function stopNodeInteraction(element) {
    for (const eventName of ["pointerdown", "mousedown", "touchstart"]) {
        element.addEventListener(eventName, (event) => event.stopPropagation());
    }
}

function syncWidget(node, name, value) {
    const widget = node.widgets?.find((candidate) => candidate.name === name);
    if (!widget) return;
    widget.value = value;
    widget.callback?.(value);
}

function hideWidget(node, name) {
    const widget = node.widgets?.find((candidate) => candidate.name === name);
    if (!widget) return;
    widget.draw = () => {};
    widget.computeSize = () => [0, -4];
}

function makeToggle(node, label, widgetName) {
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(node.widgets?.find((w) => w.name === widgetName)?.value);
    input.addEventListener("change", () => syncWidget(node, widgetName, input.checked));
    const wrapper = document.createElement("label");
    wrapper.style.cssText = "display:flex;align-items:center;gap:3px;white-space:nowrap;cursor:pointer";
    wrapper.append(input, ` ${label}`);
    return { input, wrapper };
}

app.registerExtension({
    name: "Chill.EnhancedVideoCombinePreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        const originalOnExecuted = nodeType.prototype.onExecuted;

        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated ? originalOnNodeCreated.apply(this, arguments) : undefined;
            hideWidget(this, "save_first_frame");
            hideWidget(this, "save_last_frame");

            const node = this;
            const root = document.createElement("div");
            root.style.width = "100%";

            const video = document.createElement("video");
            video.controls = true;
            video.loop = true;
            video.muted = true;
            video.playsInline = true;
            video.style.cssText = "display:block;width:100%;background:#111;cursor:pointer";

            const toolbar = document.createElement("div");
            toolbar.style.cssText =
                "display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin-top:6px;padding:5px;" +
                "background:rgba(0,0,0,.25);border:1px solid rgba(255,255,255,.08);border-radius:4px;" +
                "font:11px sans-serif;color:var(--input-text,#ddd)";

            const { wrapper: saveFirstWrap } = makeToggle(node, "Save first frame", "save_first_frame");
            const { wrapper: saveLastWrap } = makeToggle(node, "Save last frame", "save_last_frame");

            const autoplay = document.createElement("input");
            autoplay.type = "checkbox";
            autoplay.checked = true;
            const autoplayWrap = document.createElement("label");
            autoplayWrap.style.cssText = "display:flex;align-items:center;gap:3px;margin-left:auto;white-space:nowrap;cursor:pointer";
            autoplayWrap.append(autoplay, " Autoplay");

            const download = document.createElement("a");
            download.textContent = "Download";
            download.href = "#";
            download.style.cssText =
                "padding:2px 8px;color:inherit;background:rgba(255,255,255,.1);" +
                "border:1px solid rgba(255,255,255,.2);border-radius:3px;text-decoration:none;cursor:pointer";
            download.addEventListener("click", (event) => {
                if (download.getAttribute("href") === "#") event.preventDefault();
            });

            toolbar.append(saveFirstWrap, saveLastWrap, autoplayWrap, download);
            [video, toolbar, saveFirstWrap, saveLastWrap, autoplayWrap, download].forEach(stopNodeInteraction);
            root.append(video, toolbar);

            let widget;
            const previewHeight = () => (node.size[0] - 20) / (widget?.aspectRatio ?? 16 / 9) + 40;
            widget = this.addDOMWidget("chill_video_preview", "preview", root, {
                serialize: false,
                hideOnZoom: false,
                getHeight: () => previewHeight(),
            });
            widget.aspectRatio = 16 / 9;
            widget.computeSize = (width) => [width, previewHeight()];

            video.addEventListener("loadedmetadata", () => {
                widget.aspectRatio = video.videoWidth / video.videoHeight;
                node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]]);
                node.graph?.setDirtyCanvas(true);
                if (autoplay.checked) video.play().catch(() => {});
            });

            node.chillVideoPreview = video;
            node.chillVideoDownload = download;
            return result;
        };

        nodeType.prototype.onExecuted = function (message) {
            const result = originalOnExecuted ? originalOnExecuted.apply(this, arguments) : undefined;
            const video = message?.gifs?.[0];
            if (!video?.filename || !this.chillVideoPreview) return result;

            const url = assetUrl(video);
            this.chillVideoPreview.pause();
            this.chillVideoPreview.src = url;
            this.chillVideoPreview.load();
            if (this.chillVideoDownload) {
                this.chillVideoDownload.href = url;
                this.chillVideoDownload.download = video.filename;
            }
            return result;
        };
    },
});
