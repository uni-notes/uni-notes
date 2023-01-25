'use strict';

var obsidian = require('obsidian');
var state = require('@codemirror/state');
var view = require('@codemirror/view');

/*! *****************************************************************************
Copyright (c) Microsoft Corporation.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
***************************************************************************** */
/* global Reflect, Promise */

var extendStatics = function(d, b) {
    extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
    return extendStatics(d, b);
};

function __extends(d, b) {
    if (typeof b !== "function" && b !== null)
        throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
    extendStatics(d, b);
    function __() { this.constructor = d; }
    d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
}

function __awaiter(thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
}

function __generator(thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
}

// all credit to azu: https://github.com/azu/codemirror-typewriter-scrolling/blob/b0ac076d72c9445c96182de87d974de2e8cc56e2/typewriter-scrolling.js
var movedByMouse = false;
CodeMirror.commands.scrollSelectionToCenter = function (cm) {
    var cursor = cm.getCursor('head');
    var charCoords = cm.charCoords(cursor, "local");
    var top = charCoords.top;
    var halfLineHeight = (charCoords.bottom - top) / 2;
    var halfWindowHeight = cm.getWrapperElement().offsetHeight / 2;
    var scrollTo = Math.round((top - halfWindowHeight + halfLineHeight));
    cm.scrollTo(null, scrollTo);
};
CodeMirror.defineOption("typewriterScrolling", false, function (cm, val, old) {
    if (old && old != CodeMirror.Init) {
        const linesEl = cm.getScrollerElement().querySelector('.CodeMirror-lines');
        linesEl.style.paddingTop = null;
        linesEl.style.paddingBottom = null;
        cm.off("cursorActivity", onCursorActivity);
        cm.off("refresh", onRefresh);
        cm.off("mousedown", onMouseDown);
        cm.off("keydown", onKeyDown);
        cm.off("beforeChange", onBeforeChange);
    }
    if (val) {
        onRefresh(cm);
        cm.on("cursorActivity", onCursorActivity);
        cm.on("refresh", onRefresh);
        cm.on("mousedown", onMouseDown);
        cm.on("keydown", onKeyDown);
        cm.on("beforeChange", onBeforeChange);
    }
});
function onMouseDown() {
    movedByMouse = true;
}
const modiferKeys = ["Alt", "AltGraph", "CapsLock", "Control", "Fn", "FnLock", "Hyper", "Meta", "NumLock", "ScrollLock", "Shift", "Super", "Symbol", "SymbolLock"];
function onKeyDown(cm, e) {
    if (!modiferKeys.includes(e.key)) {
        movedByMouse = false;
    }
}
function onBeforeChange() {
    movedByMouse = false;
}
function onCursorActivity(cm) {
    const linesEl = cm.getScrollerElement().querySelector('.CodeMirror-lines');
    if (cm.getSelection().length !== 0) {
        linesEl.classList.add("selecting");
    }
    else {
        linesEl.classList.remove("selecting");
    }

    if(!movedByMouse) {
        cm.execCommand("scrollSelectionToCenter");
    }
}
function onRefresh(cm) {
    const halfWindowHeight = cm.getWrapperElement().offsetHeight / 2;
    const linesEl = cm.getScrollerElement().querySelector('.CodeMirror-lines');
    linesEl.style.paddingTop = `${halfWindowHeight}px`;
    linesEl.style.paddingBottom = `${halfWindowHeight}px`; // Thanks @walulula!
    if (cm.getSelection().length === 0) {
        cm.execCommand("scrollSelectionToCenter");
    }
}

var allowedUserEvents = /^(select|input|delete|undo|redo)(\..+)?$/;
var disallowedUserEvents = /^(select.pointer)$/;
var typewriterScrollPlugin = view.ViewPlugin.fromClass(/** @class */ (function () {
    function class_1(view) {
        this.view = view;
        this.myUpdate = false;
        this.padding = null;
    }
    class_1.prototype.update = function (update) {
        if (this.myUpdate)
            this.myUpdate = false;
        else {
            this.padding = ((this.view.dom.clientHeight / 2) - (this.view.defaultLineHeight)) + "px";
            if (this.padding != this.view.contentDOM.style.paddingTop) {
                this.view.contentDOM.style.paddingTop = this.view.contentDOM.style.paddingBottom = this.padding;
            }
            var userEvents = update.transactions.map(function (tr) { return tr.annotation(state.Transaction.userEvent); });
            var isAllowed = userEvents.reduce(function (result, event) { return result && allowedUserEvents.test(event) && !disallowedUserEvents.test(event); }, userEvents.length > 0);
            if (isAllowed) {
                this.myUpdate = true;
                this.centerOnHead(update);
            }
        }
    };
    class_1.prototype.centerOnHead = function (update) {
        return __awaiter(this, void 0, void 0, function () {
            var _this = this;
            return __generator(this, function (_a) {
                // can't update inside an update, so request the next animation frame
                window.requestAnimationFrame(function () {
                    // current selection range
                    if (update.state.selection.ranges.length == 1) {
                        // only act on the one range
                        var head = update.state.selection.main.head;
                        var prevHead = update.startState.selection.main.head;
                        // TODO: work out line number and use that instead? Is that even possible?
                        // don't bother with this next part if the range (line??) hasn't changed
                        if (prevHead != head) {
                            // this is the effect that does the centering
                            var effect = view.EditorView.scrollIntoView(head, { y: "center" });
                            var transaction = _this.view.state.update({ effects: effect });
                            _this.view.dispatch(transaction);
                        }
                    }
                });
                return [2 /*return*/];
            });
        });
    };
    return class_1;
}()));
function typewriterScroll(options) {
    return [
        typewriterScrollPlugin
    ];
}

var DEFAULT_SETTINGS = {
    enabled: true,
    zenEnabled: false
};
var CMTypewriterScrollPlugin = /** @class */ (function (_super) {
    __extends(CMTypewriterScrollPlugin, _super);
    function CMTypewriterScrollPlugin() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.toggleTypewriterScroll = function (newValue) {
            if (newValue === void 0) { newValue = null; }
            // if no value is passed in, toggle the existing value
            if (newValue === null)
                newValue = !_this.settings.enabled;
            // assign the new value and call the correct enable / disable function
            (_this.settings.enabled = newValue)
                ? _this.enableTypewriterScroll() : _this.disableTypewriterScroll();
            // save the new settings
            _this.saveData(_this.settings);
        };
        _this.toggleZen = function (newValue) {
            if (newValue === void 0) { newValue = null; }
            // if no value is passed in, toggle the existing value
            if (newValue === null)
                newValue = !_this.settings.zenEnabled;
            // assign the new value and call the correct enable / disable function
            (_this.settings.zenEnabled = newValue)
                ? _this.enableZen() : _this.disableZen();
            // save the new settings
            _this.saveData(_this.settings);
        };
        _this.enableTypewriterScroll = function () {
            // add the class
            document.body.classList.add('plugin-cm-typewriter-scroll');
            // register the codemirror add on setting
            _this.registerCodeMirror(function (cm) {
                // @ts-ignore
                cm.setOption("typewriterScrolling", true);
            });
            _this.registerEditorExtension(typewriterScroll());
        };
        _this.disableTypewriterScroll = function () {
            // remove the class
            document.body.classList.remove('plugin-cm-typewriter-scroll');
            // remove the codemirror add on setting
            _this.app.workspace.iterateCodeMirrors(function (cm) {
                // @ts-ignore
                cm.setOption("typewriterScrolling", false);
            });
        };
        _this.enableZen = function () {
            // add the class
            document.body.classList.add('plugin-cm-typewriter-scroll-zen');
        };
        _this.disableZen = function () {
            // remove the class
            document.body.classList.remove('plugin-cm-typewriter-scroll-zen');
        };
        return _this;
    }
    CMTypewriterScrollPlugin.prototype.onload = function () {
        return __awaiter(this, void 0, void 0, function () {
            var _a, _b, _c, _d;
            return __generator(this, function (_e) {
                switch (_e.label) {
                    case 0:
                        _a = this;
                        _c = (_b = Object).assign;
                        _d = [DEFAULT_SETTINGS];
                        return [4 /*yield*/, this.loadData()];
                    case 1:
                        _a.settings = _c.apply(_b, _d.concat([_e.sent()]));
                        // enable the plugin (based on settings)
                        if (this.settings.enabled)
                            this.enableTypewriterScroll();
                        if (this.settings.zenEnabled)
                            this.enableZen();
                        // add the settings tab
                        this.addSettingTab(new CMTypewriterScrollSettingTab(this.app, this));
                        // add the commands / keyboard shortcuts
                        this.addCommands();
                        return [2 /*return*/];
                }
            });
        });
    };
    CMTypewriterScrollPlugin.prototype.onunload = function () {
        // disable the plugin
        this.disableTypewriterScroll();
        this.disableZen();
    };
    CMTypewriterScrollPlugin.prototype.addCommands = function () {
        var _this = this;
        // add the toggle on/off command
        this.addCommand({
            id: 'toggle-typewriter-sroll',
            name: 'Toggle On/Off',
            callback: function () { _this.toggleTypewriterScroll(); }
        });
        // toggle zen mode
        this.addCommand({
            id: 'toggle-typewriter-sroll-zen',
            name: 'Toggle Zen Mode On/Off',
            callback: function () { _this.toggleZen(); }
        });
    };
    return CMTypewriterScrollPlugin;
}(obsidian.Plugin));
var CMTypewriterScrollSettingTab = /** @class */ (function (_super) {
    __extends(CMTypewriterScrollSettingTab, _super);
    function CMTypewriterScrollSettingTab(app, plugin) {
        var _this = _super.call(this, app, plugin) || this;
        _this.plugin = plugin;
        return _this;
    }
    CMTypewriterScrollSettingTab.prototype.display = function () {
        var _this = this;
        var containerEl = this.containerEl;
        containerEl.empty();
        new obsidian.Setting(containerEl)
            .setName("Toggle Typewriter Scrolling")
            .setDesc("Turns typewriter scrolling on or off globally")
            .addToggle(function (toggle) {
            return toggle.setValue(_this.plugin.settings.enabled)
                .onChange(function (newValue) { _this.plugin.toggleTypewriterScroll(newValue); });
        });
        new obsidian.Setting(containerEl)
            .setName("Zen Mode")
            .setDesc("Darkens non-active lines in the editor so you can focus on what you're typing")
            .addToggle(function (toggle) {
            return toggle.setValue(_this.plugin.settings.zenEnabled)
                .onChange(function (newValue) { _this.plugin.toggleZen(newValue); });
        });
    };
    return CMTypewriterScrollSettingTab;
}(obsidian.PluginSettingTab));

module.exports = CMTypewriterScrollPlugin;
//## sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibWFpbi5qcyIsInNvdXJjZXMiOlsibm9kZV9tb2R1bGVzL3RzbGliL3RzbGliLmVzNi5qcyIsInR5cGV3cml0ZXItc2Nyb2xsaW5nLmpzIiwiZXh0ZW5zaW9uLnRzIiwibWFpbi50cyJdLCJzb3VyY2VzQ29udGVudCI6WyIvKiEgKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKipcclxuQ29weXJpZ2h0IChjKSBNaWNyb3NvZnQgQ29ycG9yYXRpb24uXHJcblxyXG5QZXJtaXNzaW9uIHRvIHVzZSwgY29weSwgbW9kaWZ5LCBhbmQvb3IgZGlzdHJpYnV0ZSB0aGlzIHNvZnR3YXJlIGZvciBhbnlcclxucHVycG9zZSB3aXRoIG9yIHdpdGhvdXQgZmVlIGlzIGhlcmVieSBncmFudGVkLlxyXG5cclxuVEhFIFNPRlRXQVJFIElTIFBST1ZJREVEIFwiQVMgSVNcIiBBTkQgVEhFIEFVVEhPUiBESVNDTEFJTVMgQUxMIFdBUlJBTlRJRVMgV0lUSFxyXG5SRUdBUkQgVE8gVEhJUyBTT0ZUV0FSRSBJTkNMVURJTkcgQUxMIElNUExJRUQgV0FSUkFOVElFUyBPRiBNRVJDSEFOVEFCSUxJVFlcclxuQU5EIEZJVE5FU1MuIElOIE5PIEVWRU5UIFNIQUxMIFRIRSBBVVRIT1IgQkUgTElBQkxFIEZPUiBBTlkgU1BFQ0lBTCwgRElSRUNULFxyXG5JTkRJUkVDVCwgT1IgQ09OU0VRVUVOVElBTCBEQU1BR0VTIE9SIEFOWSBEQU1BR0VTIFdIQVRTT0VWRVIgUkVTVUxUSU5HIEZST01cclxuTE9TUyBPRiBVU0UsIERBVEEgT1IgUFJPRklUUywgV0hFVEhFUiBJTiBBTiBBQ1RJT04gT0YgQ09OVFJBQ1QsIE5FR0xJR0VOQ0UgT1JcclxuT1RIRVIgVE9SVElPVVMgQUNUSU9OLCBBUklTSU5HIE9VVCBPRiBPUiBJTiBDT05ORUNUSU9OIFdJVEggVEhFIFVTRSBPUlxyXG5QRVJGT1JNQU5DRSBPRiBUSElTIFNPRlRXQVJFLlxyXG4qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKiAqL1xyXG4vKiBnbG9iYWwgUmVmbGVjdCwgUHJvbWlzZSAqL1xyXG5cclxudmFyIGV4dGVuZFN0YXRpY3MgPSBmdW5jdGlvbihkLCBiKSB7XHJcbiAgICBleHRlbmRTdGF0aWNzID0gT2JqZWN0LnNldFByb3RvdHlwZU9mIHx8XHJcbiAgICAgICAgKHsgX19wcm90b19fOiBbXSB9IGluc3RhbmNlb2YgQXJyYXkgJiYgZnVuY3Rpb24gKGQsIGIpIHsgZC5fX3Byb3RvX18gPSBiOyB9KSB8fFxyXG4gICAgICAgIGZ1bmN0aW9uIChkLCBiKSB7IGZvciAodmFyIHAgaW4gYikgaWYgKE9iamVjdC5wcm90b3R5cGUuaGFzT3duUHJvcGVydHkuY2FsbChiLCBwKSkgZFtwXSA9IGJbcF07IH07XHJcbiAgICByZXR1cm4gZXh0ZW5kU3RhdGljcyhkLCBiKTtcclxufTtcclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX2V4dGVuZHMoZCwgYikge1xyXG4gICAgaWYgKHR5cGVvZiBiICE9PSBcImZ1bmN0aW9uXCIgJiYgYiAhPT0gbnVsbClcclxuICAgICAgICB0aHJvdyBuZXcgVHlwZUVycm9yKFwiQ2xhc3MgZXh0ZW5kcyB2YWx1ZSBcIiArIFN0cmluZyhiKSArIFwiIGlzIG5vdCBhIGNvbnN0cnVjdG9yIG9yIG51bGxcIik7XHJcbiAgICBleHRlbmRTdGF0aWNzKGQsIGIpO1xyXG4gICAgZnVuY3Rpb24gX18oKSB7IHRoaXMuY29uc3RydWN0b3IgPSBkOyB9XHJcbiAgICBkLnByb3RvdHlwZSA9IGIgPT09IG51bGwgPyBPYmplY3QuY3JlYXRlKGIpIDogKF9fLnByb3RvdHlwZSA9IGIucHJvdG90eXBlLCBuZXcgX18oKSk7XHJcbn1cclxuXHJcbmV4cG9ydCB2YXIgX19hc3NpZ24gPSBmdW5jdGlvbigpIHtcclxuICAgIF9fYXNzaWduID0gT2JqZWN0LmFzc2lnbiB8fCBmdW5jdGlvbiBfX2Fzc2lnbih0KSB7XHJcbiAgICAgICAgZm9yICh2YXIgcywgaSA9IDEsIG4gPSBhcmd1bWVudHMubGVuZ3RoOyBpIDwgbjsgaSsrKSB7XHJcbiAgICAgICAgICAgIHMgPSBhcmd1bWVudHNbaV07XHJcbiAgICAgICAgICAgIGZvciAodmFyIHAgaW4gcykgaWYgKE9iamVjdC5wcm90b3R5cGUuaGFzT3duUHJvcGVydHkuY2FsbChzLCBwKSkgdFtwXSA9IHNbcF07XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiB0O1xyXG4gICAgfVxyXG4gICAgcmV0dXJuIF9fYXNzaWduLmFwcGx5KHRoaXMsIGFyZ3VtZW50cyk7XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX3Jlc3QocywgZSkge1xyXG4gICAgdmFyIHQgPSB7fTtcclxuICAgIGZvciAodmFyIHAgaW4gcykgaWYgKE9iamVjdC5wcm90b3R5cGUuaGFzT3duUHJvcGVydHkuY2FsbChzLCBwKSAmJiBlLmluZGV4T2YocCkgPCAwKVxyXG4gICAgICAgIHRbcF0gPSBzW3BdO1xyXG4gICAgaWYgKHMgIT0gbnVsbCAmJiB0eXBlb2YgT2JqZWN0LmdldE93blByb3BlcnR5U3ltYm9scyA9PT0gXCJmdW5jdGlvblwiKVxyXG4gICAgICAgIGZvciAodmFyIGkgPSAwLCBwID0gT2JqZWN0LmdldE93blByb3BlcnR5U3ltYm9scyhzKTsgaSA8IHAubGVuZ3RoOyBpKyspIHtcclxuICAgICAgICAgICAgaWYgKGUuaW5kZXhPZihwW2ldKSA8IDAgJiYgT2JqZWN0LnByb3RvdHlwZS5wcm9wZXJ0eUlzRW51bWVyYWJsZS5jYWxsKHMsIHBbaV0pKVxyXG4gICAgICAgICAgICAgICAgdFtwW2ldXSA9IHNbcFtpXV07XHJcbiAgICAgICAgfVxyXG4gICAgcmV0dXJuIHQ7XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX2RlY29yYXRlKGRlY29yYXRvcnMsIHRhcmdldCwga2V5LCBkZXNjKSB7XHJcbiAgICB2YXIgYyA9IGFyZ3VtZW50cy5sZW5ndGgsIHIgPSBjIDwgMyA/IHRhcmdldCA6IGRlc2MgPT09IG51bGwgPyBkZXNjID0gT2JqZWN0LmdldE93blByb3BlcnR5RGVzY3JpcHRvcih0YXJnZXQsIGtleSkgOiBkZXNjLCBkO1xyXG4gICAgaWYgKHR5cGVvZiBSZWZsZWN0ID09PSBcIm9iamVjdFwiICYmIHR5cGVvZiBSZWZsZWN0LmRlY29yYXRlID09PSBcImZ1bmN0aW9uXCIpIHIgPSBSZWZsZWN0LmRlY29yYXRlKGRlY29yYXRvcnMsIHRhcmdldCwga2V5LCBkZXNjKTtcclxuICAgIGVsc2UgZm9yICh2YXIgaSA9IGRlY29yYXRvcnMubGVuZ3RoIC0gMTsgaSA+PSAwOyBpLS0pIGlmIChkID0gZGVjb3JhdG9yc1tpXSkgciA9IChjIDwgMyA/IGQocikgOiBjID4gMyA/IGQodGFyZ2V0LCBrZXksIHIpIDogZCh0YXJnZXQsIGtleSkpIHx8IHI7XHJcbiAgICByZXR1cm4gYyA+IDMgJiYgciAmJiBPYmplY3QuZGVmaW5lUHJvcGVydHkodGFyZ2V0LCBrZXksIHIpLCByO1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19wYXJhbShwYXJhbUluZGV4LCBkZWNvcmF0b3IpIHtcclxuICAgIHJldHVybiBmdW5jdGlvbiAodGFyZ2V0LCBrZXkpIHsgZGVjb3JhdG9yKHRhcmdldCwga2V5LCBwYXJhbUluZGV4KTsgfVxyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19tZXRhZGF0YShtZXRhZGF0YUtleSwgbWV0YWRhdGFWYWx1ZSkge1xyXG4gICAgaWYgKHR5cGVvZiBSZWZsZWN0ID09PSBcIm9iamVjdFwiICYmIHR5cGVvZiBSZWZsZWN0Lm1ldGFkYXRhID09PSBcImZ1bmN0aW9uXCIpIHJldHVybiBSZWZsZWN0Lm1ldGFkYXRhKG1ldGFkYXRhS2V5LCBtZXRhZGF0YVZhbHVlKTtcclxufVxyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fYXdhaXRlcih0aGlzQXJnLCBfYXJndW1lbnRzLCBQLCBnZW5lcmF0b3IpIHtcclxuICAgIGZ1bmN0aW9uIGFkb3B0KHZhbHVlKSB7IHJldHVybiB2YWx1ZSBpbnN0YW5jZW9mIFAgPyB2YWx1ZSA6IG5ldyBQKGZ1bmN0aW9uIChyZXNvbHZlKSB7IHJlc29sdmUodmFsdWUpOyB9KTsgfVxyXG4gICAgcmV0dXJuIG5ldyAoUCB8fCAoUCA9IFByb21pc2UpKShmdW5jdGlvbiAocmVzb2x2ZSwgcmVqZWN0KSB7XHJcbiAgICAgICAgZnVuY3Rpb24gZnVsZmlsbGVkKHZhbHVlKSB7IHRyeSB7IHN0ZXAoZ2VuZXJhdG9yLm5leHQodmFsdWUpKTsgfSBjYXRjaCAoZSkgeyByZWplY3QoZSk7IH0gfVxyXG4gICAgICAgIGZ1bmN0aW9uIHJlamVjdGVkKHZhbHVlKSB7IHRyeSB7IHN0ZXAoZ2VuZXJhdG9yW1widGhyb3dcIl0odmFsdWUpKTsgfSBjYXRjaCAoZSkgeyByZWplY3QoZSk7IH0gfVxyXG4gICAgICAgIGZ1bmN0aW9uIHN0ZXAocmVzdWx0KSB7IHJlc3VsdC5kb25lID8gcmVzb2x2ZShyZXN1bHQudmFsdWUpIDogYWRvcHQocmVzdWx0LnZhbHVlKS50aGVuKGZ1bGZpbGxlZCwgcmVqZWN0ZWQpOyB9XHJcbiAgICAgICAgc3RlcCgoZ2VuZXJhdG9yID0gZ2VuZXJhdG9yLmFwcGx5KHRoaXNBcmcsIF9hcmd1bWVudHMgfHwgW10pKS5uZXh0KCkpO1xyXG4gICAgfSk7XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX2dlbmVyYXRvcih0aGlzQXJnLCBib2R5KSB7XHJcbiAgICB2YXIgXyA9IHsgbGFiZWw6IDAsIHNlbnQ6IGZ1bmN0aW9uKCkgeyBpZiAodFswXSAmIDEpIHRocm93IHRbMV07IHJldHVybiB0WzFdOyB9LCB0cnlzOiBbXSwgb3BzOiBbXSB9LCBmLCB5LCB0LCBnO1xyXG4gICAgcmV0dXJuIGcgPSB7IG5leHQ6IHZlcmIoMCksIFwidGhyb3dcIjogdmVyYigxKSwgXCJyZXR1cm5cIjogdmVyYigyKSB9LCB0eXBlb2YgU3ltYm9sID09PSBcImZ1bmN0aW9uXCIgJiYgKGdbU3ltYm9sLml0ZXJhdG9yXSA9IGZ1bmN0aW9uKCkgeyByZXR1cm4gdGhpczsgfSksIGc7XHJcbiAgICBmdW5jdGlvbiB2ZXJiKG4pIHsgcmV0dXJuIGZ1bmN0aW9uICh2KSB7IHJldHVybiBzdGVwKFtuLCB2XSk7IH07IH1cclxuICAgIGZ1bmN0aW9uIHN0ZXAob3ApIHtcclxuICAgICAgICBpZiAoZikgdGhyb3cgbmV3IFR5cGVFcnJvcihcIkdlbmVyYXRvciBpcyBhbHJlYWR5IGV4ZWN1dGluZy5cIik7XHJcbiAgICAgICAgd2hpbGUgKF8pIHRyeSB7XHJcbiAgICAgICAgICAgIGlmIChmID0gMSwgeSAmJiAodCA9IG9wWzBdICYgMiA/IHlbXCJyZXR1cm5cIl0gOiBvcFswXSA/IHlbXCJ0aHJvd1wiXSB8fCAoKHQgPSB5W1wicmV0dXJuXCJdKSAmJiB0LmNhbGwoeSksIDApIDogeS5uZXh0KSAmJiAhKHQgPSB0LmNhbGwoeSwgb3BbMV0pKS5kb25lKSByZXR1cm4gdDtcclxuICAgICAgICAgICAgaWYgKHkgPSAwLCB0KSBvcCA9IFtvcFswXSAmIDIsIHQudmFsdWVdO1xyXG4gICAgICAgICAgICBzd2l0Y2ggKG9wWzBdKSB7XHJcbiAgICAgICAgICAgICAgICBjYXNlIDA6IGNhc2UgMTogdCA9IG9wOyBicmVhaztcclxuICAgICAgICAgICAgICAgIGNhc2UgNDogXy5sYWJlbCsrOyByZXR1cm4geyB2YWx1ZTogb3BbMV0sIGRvbmU6IGZhbHNlIH07XHJcbiAgICAgICAgICAgICAgICBjYXNlIDU6IF8ubGFiZWwrKzsgeSA9IG9wWzFdOyBvcCA9IFswXTsgY29udGludWU7XHJcbiAgICAgICAgICAgICAgICBjYXNlIDc6IG9wID0gXy5vcHMucG9wKCk7IF8udHJ5cy5wb3AoKTsgY29udGludWU7XHJcbiAgICAgICAgICAgICAgICBkZWZhdWx0OlxyXG4gICAgICAgICAgICAgICAgICAgIGlmICghKHQgPSBfLnRyeXMsIHQgPSB0Lmxlbmd0aCA+IDAgJiYgdFt0Lmxlbmd0aCAtIDFdKSAmJiAob3BbMF0gPT09IDYgfHwgb3BbMF0gPT09IDIpKSB7IF8gPSAwOyBjb250aW51ZTsgfVxyXG4gICAgICAgICAgICAgICAgICAgIGlmIChvcFswXSA9PT0gMyAmJiAoIXQgfHwgKG9wWzFdID4gdFswXSAmJiBvcFsxXSA8IHRbM10pKSkgeyBfLmxhYmVsID0gb3BbMV07IGJyZWFrOyB9XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKG9wWzBdID09PSA2ICYmIF8ubGFiZWwgPCB0WzFdKSB7IF8ubGFiZWwgPSB0WzFdOyB0ID0gb3A7IGJyZWFrOyB9XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKHQgJiYgXy5sYWJlbCA8IHRbMl0pIHsgXy5sYWJlbCA9IHRbMl07IF8ub3BzLnB1c2gob3ApOyBicmVhazsgfVxyXG4gICAgICAgICAgICAgICAgICAgIGlmICh0WzJdKSBfLm9wcy5wb3AoKTtcclxuICAgICAgICAgICAgICAgICAgICBfLnRyeXMucG9wKCk7IGNvbnRpbnVlO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIG9wID0gYm9keS5jYWxsKHRoaXNBcmcsIF8pO1xyXG4gICAgICAgIH0gY2F0Y2ggKGUpIHsgb3AgPSBbNiwgZV07IHkgPSAwOyB9IGZpbmFsbHkgeyBmID0gdCA9IDA7IH1cclxuICAgICAgICBpZiAob3BbMF0gJiA1KSB0aHJvdyBvcFsxXTsgcmV0dXJuIHsgdmFsdWU6IG9wWzBdID8gb3BbMV0gOiB2b2lkIDAsIGRvbmU6IHRydWUgfTtcclxuICAgIH1cclxufVxyXG5cclxuZXhwb3J0IHZhciBfX2NyZWF0ZUJpbmRpbmcgPSBPYmplY3QuY3JlYXRlID8gKGZ1bmN0aW9uKG8sIG0sIGssIGsyKSB7XHJcbiAgICBpZiAoazIgPT09IHVuZGVmaW5lZCkgazIgPSBrO1xyXG4gICAgT2JqZWN0LmRlZmluZVByb3BlcnR5KG8sIGsyLCB7IGVudW1lcmFibGU6IHRydWUsIGdldDogZnVuY3Rpb24oKSB7IHJldHVybiBtW2tdOyB9IH0pO1xyXG59KSA6IChmdW5jdGlvbihvLCBtLCBrLCBrMikge1xyXG4gICAgaWYgKGsyID09PSB1bmRlZmluZWQpIGsyID0gaztcclxuICAgIG9bazJdID0gbVtrXTtcclxufSk7XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19leHBvcnRTdGFyKG0sIG8pIHtcclxuICAgIGZvciAodmFyIHAgaW4gbSkgaWYgKHAgIT09IFwiZGVmYXVsdFwiICYmICFPYmplY3QucHJvdG90eXBlLmhhc093blByb3BlcnR5LmNhbGwobywgcCkpIF9fY3JlYXRlQmluZGluZyhvLCBtLCBwKTtcclxufVxyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fdmFsdWVzKG8pIHtcclxuICAgIHZhciBzID0gdHlwZW9mIFN5bWJvbCA9PT0gXCJmdW5jdGlvblwiICYmIFN5bWJvbC5pdGVyYXRvciwgbSA9IHMgJiYgb1tzXSwgaSA9IDA7XHJcbiAgICBpZiAobSkgcmV0dXJuIG0uY2FsbChvKTtcclxuICAgIGlmIChvICYmIHR5cGVvZiBvLmxlbmd0aCA9PT0gXCJudW1iZXJcIikgcmV0dXJuIHtcclxuICAgICAgICBuZXh0OiBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgICAgIGlmIChvICYmIGkgPj0gby5sZW5ndGgpIG8gPSB2b2lkIDA7XHJcbiAgICAgICAgICAgIHJldHVybiB7IHZhbHVlOiBvICYmIG9baSsrXSwgZG9uZTogIW8gfTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgdGhyb3cgbmV3IFR5cGVFcnJvcihzID8gXCJPYmplY3QgaXMgbm90IGl0ZXJhYmxlLlwiIDogXCJTeW1ib2wuaXRlcmF0b3IgaXMgbm90IGRlZmluZWQuXCIpO1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19yZWFkKG8sIG4pIHtcclxuICAgIHZhciBtID0gdHlwZW9mIFN5bWJvbCA9PT0gXCJmdW5jdGlvblwiICYmIG9bU3ltYm9sLml0ZXJhdG9yXTtcclxuICAgIGlmICghbSkgcmV0dXJuIG87XHJcbiAgICB2YXIgaSA9IG0uY2FsbChvKSwgciwgYXIgPSBbXSwgZTtcclxuICAgIHRyeSB7XHJcbiAgICAgICAgd2hpbGUgKChuID09PSB2b2lkIDAgfHwgbi0tID4gMCkgJiYgIShyID0gaS5uZXh0KCkpLmRvbmUpIGFyLnB1c2goci52YWx1ZSk7XHJcbiAgICB9XHJcbiAgICBjYXRjaCAoZXJyb3IpIHsgZSA9IHsgZXJyb3I6IGVycm9yIH07IH1cclxuICAgIGZpbmFsbHkge1xyXG4gICAgICAgIHRyeSB7XHJcbiAgICAgICAgICAgIGlmIChyICYmICFyLmRvbmUgJiYgKG0gPSBpW1wicmV0dXJuXCJdKSkgbS5jYWxsKGkpO1xyXG4gICAgICAgIH1cclxuICAgICAgICBmaW5hbGx5IHsgaWYgKGUpIHRocm93IGUuZXJyb3I7IH1cclxuICAgIH1cclxuICAgIHJldHVybiBhcjtcclxufVxyXG5cclxuLyoqIEBkZXByZWNhdGVkICovXHJcbmV4cG9ydCBmdW5jdGlvbiBfX3NwcmVhZCgpIHtcclxuICAgIGZvciAodmFyIGFyID0gW10sIGkgPSAwOyBpIDwgYXJndW1lbnRzLmxlbmd0aDsgaSsrKVxyXG4gICAgICAgIGFyID0gYXIuY29uY2F0KF9fcmVhZChhcmd1bWVudHNbaV0pKTtcclxuICAgIHJldHVybiBhcjtcclxufVxyXG5cclxuLyoqIEBkZXByZWNhdGVkICovXHJcbmV4cG9ydCBmdW5jdGlvbiBfX3NwcmVhZEFycmF5cygpIHtcclxuICAgIGZvciAodmFyIHMgPSAwLCBpID0gMCwgaWwgPSBhcmd1bWVudHMubGVuZ3RoOyBpIDwgaWw7IGkrKykgcyArPSBhcmd1bWVudHNbaV0ubGVuZ3RoO1xyXG4gICAgZm9yICh2YXIgciA9IEFycmF5KHMpLCBrID0gMCwgaSA9IDA7IGkgPCBpbDsgaSsrKVxyXG4gICAgICAgIGZvciAodmFyIGEgPSBhcmd1bWVudHNbaV0sIGogPSAwLCBqbCA9IGEubGVuZ3RoOyBqIDwgamw7IGorKywgaysrKVxyXG4gICAgICAgICAgICByW2tdID0gYVtqXTtcclxuICAgIHJldHVybiByO1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19zcHJlYWRBcnJheSh0bywgZnJvbSwgcGFjaykge1xyXG4gICAgaWYgKHBhY2sgfHwgYXJndW1lbnRzLmxlbmd0aCA9PT0gMikgZm9yICh2YXIgaSA9IDAsIGwgPSBmcm9tLmxlbmd0aCwgYXI7IGkgPCBsOyBpKyspIHtcclxuICAgICAgICBpZiAoYXIgfHwgIShpIGluIGZyb20pKSB7XHJcbiAgICAgICAgICAgIGlmICghYXIpIGFyID0gQXJyYXkucHJvdG90eXBlLnNsaWNlLmNhbGwoZnJvbSwgMCwgaSk7XHJcbiAgICAgICAgICAgIGFyW2ldID0gZnJvbVtpXTtcclxuICAgICAgICB9XHJcbiAgICB9XHJcbiAgICByZXR1cm4gdG8uY29uY2F0KGFyIHx8IEFycmF5LnByb3RvdHlwZS5zbGljZS5jYWxsKGZyb20pKTtcclxufVxyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fYXdhaXQodikge1xyXG4gICAgcmV0dXJuIHRoaXMgaW5zdGFuY2VvZiBfX2F3YWl0ID8gKHRoaXMudiA9IHYsIHRoaXMpIDogbmV3IF9fYXdhaXQodik7XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX2FzeW5jR2VuZXJhdG9yKHRoaXNBcmcsIF9hcmd1bWVudHMsIGdlbmVyYXRvcikge1xyXG4gICAgaWYgKCFTeW1ib2wuYXN5bmNJdGVyYXRvcikgdGhyb3cgbmV3IFR5cGVFcnJvcihcIlN5bWJvbC5hc3luY0l0ZXJhdG9yIGlzIG5vdCBkZWZpbmVkLlwiKTtcclxuICAgIHZhciBnID0gZ2VuZXJhdG9yLmFwcGx5KHRoaXNBcmcsIF9hcmd1bWVudHMgfHwgW10pLCBpLCBxID0gW107XHJcbiAgICByZXR1cm4gaSA9IHt9LCB2ZXJiKFwibmV4dFwiKSwgdmVyYihcInRocm93XCIpLCB2ZXJiKFwicmV0dXJuXCIpLCBpW1N5bWJvbC5hc3luY0l0ZXJhdG9yXSA9IGZ1bmN0aW9uICgpIHsgcmV0dXJuIHRoaXM7IH0sIGk7XHJcbiAgICBmdW5jdGlvbiB2ZXJiKG4pIHsgaWYgKGdbbl0pIGlbbl0gPSBmdW5jdGlvbiAodikgeyByZXR1cm4gbmV3IFByb21pc2UoZnVuY3Rpb24gKGEsIGIpIHsgcS5wdXNoKFtuLCB2LCBhLCBiXSkgPiAxIHx8IHJlc3VtZShuLCB2KTsgfSk7IH07IH1cclxuICAgIGZ1bmN0aW9uIHJlc3VtZShuLCB2KSB7IHRyeSB7IHN0ZXAoZ1tuXSh2KSk7IH0gY2F0Y2ggKGUpIHsgc2V0dGxlKHFbMF1bM10sIGUpOyB9IH1cclxuICAgIGZ1bmN0aW9uIHN0ZXAocikgeyByLnZhbHVlIGluc3RhbmNlb2YgX19hd2FpdCA/IFByb21pc2UucmVzb2x2ZShyLnZhbHVlLnYpLnRoZW4oZnVsZmlsbCwgcmVqZWN0KSA6IHNldHRsZShxWzBdWzJdLCByKTsgfVxyXG4gICAgZnVuY3Rpb24gZnVsZmlsbCh2YWx1ZSkgeyByZXN1bWUoXCJuZXh0XCIsIHZhbHVlKTsgfVxyXG4gICAgZnVuY3Rpb24gcmVqZWN0KHZhbHVlKSB7IHJlc3VtZShcInRocm93XCIsIHZhbHVlKTsgfVxyXG4gICAgZnVuY3Rpb24gc2V0dGxlKGYsIHYpIHsgaWYgKGYodiksIHEuc2hpZnQoKSwgcS5sZW5ndGgpIHJlc3VtZShxWzBdWzBdLCBxWzBdWzFdKTsgfVxyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19hc3luY0RlbGVnYXRvcihvKSB7XHJcbiAgICB2YXIgaSwgcDtcclxuICAgIHJldHVybiBpID0ge30sIHZlcmIoXCJuZXh0XCIpLCB2ZXJiKFwidGhyb3dcIiwgZnVuY3Rpb24gKGUpIHsgdGhyb3cgZTsgfSksIHZlcmIoXCJyZXR1cm5cIiksIGlbU3ltYm9sLml0ZXJhdG9yXSA9IGZ1bmN0aW9uICgpIHsgcmV0dXJuIHRoaXM7IH0sIGk7XHJcbiAgICBmdW5jdGlvbiB2ZXJiKG4sIGYpIHsgaVtuXSA9IG9bbl0gPyBmdW5jdGlvbiAodikgeyByZXR1cm4gKHAgPSAhcCkgPyB7IHZhbHVlOiBfX2F3YWl0KG9bbl0odikpLCBkb25lOiBuID09PSBcInJldHVyblwiIH0gOiBmID8gZih2KSA6IHY7IH0gOiBmOyB9XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX2FzeW5jVmFsdWVzKG8pIHtcclxuICAgIGlmICghU3ltYm9sLmFzeW5jSXRlcmF0b3IpIHRocm93IG5ldyBUeXBlRXJyb3IoXCJTeW1ib2wuYXN5bmNJdGVyYXRvciBpcyBub3QgZGVmaW5lZC5cIik7XHJcbiAgICB2YXIgbSA9IG9bU3ltYm9sLmFzeW5jSXRlcmF0b3JdLCBpO1xyXG4gICAgcmV0dXJuIG0gPyBtLmNhbGwobykgOiAobyA9IHR5cGVvZiBfX3ZhbHVlcyA9PT0gXCJmdW5jdGlvblwiID8gX192YWx1ZXMobykgOiBvW1N5bWJvbC5pdGVyYXRvcl0oKSwgaSA9IHt9LCB2ZXJiKFwibmV4dFwiKSwgdmVyYihcInRocm93XCIpLCB2ZXJiKFwicmV0dXJuXCIpLCBpW1N5bWJvbC5hc3luY0l0ZXJhdG9yXSA9IGZ1bmN0aW9uICgpIHsgcmV0dXJuIHRoaXM7IH0sIGkpO1xyXG4gICAgZnVuY3Rpb24gdmVyYihuKSB7IGlbbl0gPSBvW25dICYmIGZ1bmN0aW9uICh2KSB7IHJldHVybiBuZXcgUHJvbWlzZShmdW5jdGlvbiAocmVzb2x2ZSwgcmVqZWN0KSB7IHYgPSBvW25dKHYpLCBzZXR0bGUocmVzb2x2ZSwgcmVqZWN0LCB2LmRvbmUsIHYudmFsdWUpOyB9KTsgfTsgfVxyXG4gICAgZnVuY3Rpb24gc2V0dGxlKHJlc29sdmUsIHJlamVjdCwgZCwgdikgeyBQcm9taXNlLnJlc29sdmUodikudGhlbihmdW5jdGlvbih2KSB7IHJlc29sdmUoeyB2YWx1ZTogdiwgZG9uZTogZCB9KTsgfSwgcmVqZWN0KTsgfVxyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19tYWtlVGVtcGxhdGVPYmplY3QoY29va2VkLCByYXcpIHtcclxuICAgIGlmIChPYmplY3QuZGVmaW5lUHJvcGVydHkpIHsgT2JqZWN0LmRlZmluZVByb3BlcnR5KGNvb2tlZCwgXCJyYXdcIiwgeyB2YWx1ZTogcmF3IH0pOyB9IGVsc2UgeyBjb29rZWQucmF3ID0gcmF3OyB9XHJcbiAgICByZXR1cm4gY29va2VkO1xyXG59O1xyXG5cclxudmFyIF9fc2V0TW9kdWxlRGVmYXVsdCA9IE9iamVjdC5jcmVhdGUgPyAoZnVuY3Rpb24obywgdikge1xyXG4gICAgT2JqZWN0LmRlZmluZVByb3BlcnR5KG8sIFwiZGVmYXVsdFwiLCB7IGVudW1lcmFibGU6IHRydWUsIHZhbHVlOiB2IH0pO1xyXG59KSA6IGZ1bmN0aW9uKG8sIHYpIHtcclxuICAgIG9bXCJkZWZhdWx0XCJdID0gdjtcclxufTtcclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX2ltcG9ydFN0YXIobW9kKSB7XHJcbiAgICBpZiAobW9kICYmIG1vZC5fX2VzTW9kdWxlKSByZXR1cm4gbW9kO1xyXG4gICAgdmFyIHJlc3VsdCA9IHt9O1xyXG4gICAgaWYgKG1vZCAhPSBudWxsKSBmb3IgKHZhciBrIGluIG1vZCkgaWYgKGsgIT09IFwiZGVmYXVsdFwiICYmIE9iamVjdC5wcm90b3R5cGUuaGFzT3duUHJvcGVydHkuY2FsbChtb2QsIGspKSBfX2NyZWF0ZUJpbmRpbmcocmVzdWx0LCBtb2QsIGspO1xyXG4gICAgX19zZXRNb2R1bGVEZWZhdWx0KHJlc3VsdCwgbW9kKTtcclxuICAgIHJldHVybiByZXN1bHQ7XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX2ltcG9ydERlZmF1bHQobW9kKSB7XHJcbiAgICByZXR1cm4gKG1vZCAmJiBtb2QuX19lc01vZHVsZSkgPyBtb2QgOiB7IGRlZmF1bHQ6IG1vZCB9O1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19jbGFzc1ByaXZhdGVGaWVsZEdldChyZWNlaXZlciwgc3RhdGUsIGtpbmQsIGYpIHtcclxuICAgIGlmIChraW5kID09PSBcImFcIiAmJiAhZikgdGhyb3cgbmV3IFR5cGVFcnJvcihcIlByaXZhdGUgYWNjZXNzb3Igd2FzIGRlZmluZWQgd2l0aG91dCBhIGdldHRlclwiKTtcclxuICAgIGlmICh0eXBlb2Ygc3RhdGUgPT09IFwiZnVuY3Rpb25cIiA/IHJlY2VpdmVyICE9PSBzdGF0ZSB8fCAhZiA6ICFzdGF0ZS5oYXMocmVjZWl2ZXIpKSB0aHJvdyBuZXcgVHlwZUVycm9yKFwiQ2Fubm90IHJlYWQgcHJpdmF0ZSBtZW1iZXIgZnJvbSBhbiBvYmplY3Qgd2hvc2UgY2xhc3MgZGlkIG5vdCBkZWNsYXJlIGl0XCIpO1xyXG4gICAgcmV0dXJuIGtpbmQgPT09IFwibVwiID8gZiA6IGtpbmQgPT09IFwiYVwiID8gZi5jYWxsKHJlY2VpdmVyKSA6IGYgPyBmLnZhbHVlIDogc3RhdGUuZ2V0KHJlY2VpdmVyKTtcclxufVxyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fY2xhc3NQcml2YXRlRmllbGRTZXQocmVjZWl2ZXIsIHN0YXRlLCB2YWx1ZSwga2luZCwgZikge1xyXG4gICAgaWYgKGtpbmQgPT09IFwibVwiKSB0aHJvdyBuZXcgVHlwZUVycm9yKFwiUHJpdmF0ZSBtZXRob2QgaXMgbm90IHdyaXRhYmxlXCIpO1xyXG4gICAgaWYgKGtpbmQgPT09IFwiYVwiICYmICFmKSB0aHJvdyBuZXcgVHlwZUVycm9yKFwiUHJpdmF0ZSBhY2Nlc3NvciB3YXMgZGVmaW5lZCB3aXRob3V0IGEgc2V0dGVyXCIpO1xyXG4gICAgaWYgKHR5cGVvZiBzdGF0ZSA9PT0gXCJmdW5jdGlvblwiID8gcmVjZWl2ZXIgIT09IHN0YXRlIHx8ICFmIDogIXN0YXRlLmhhcyhyZWNlaXZlcikpIHRocm93IG5ldyBUeXBlRXJyb3IoXCJDYW5ub3Qgd3JpdGUgcHJpdmF0ZSBtZW1iZXIgdG8gYW4gb2JqZWN0IHdob3NlIGNsYXNzIGRpZCBub3QgZGVjbGFyZSBpdFwiKTtcclxuICAgIHJldHVybiAoa2luZCA9PT0gXCJhXCIgPyBmLmNhbGwocmVjZWl2ZXIsIHZhbHVlKSA6IGYgPyBmLnZhbHVlID0gdmFsdWUgOiBzdGF0ZS5zZXQocmVjZWl2ZXIsIHZhbHVlKSksIHZhbHVlO1xyXG59XHJcbiIsIi8vIGFsbCBjcmVkaXQgdG8gYXp1OiBodHRwczovL2dpdGh1Yi5jb20vYXp1L2NvZGVtaXJyb3ItdHlwZXdyaXRlci1zY3JvbGxpbmcvYmxvYi9iMGFjMDc2ZDcyYzk0NDVjOTYxODJkZTg3ZDk3NGRlMmU4Y2M1NmUyL3R5cGV3cml0ZXItc2Nyb2xsaW5nLmpzXG5cInVzZSBzdHJpY3RcIjtcbnZhciBtb3ZlZEJ5TW91c2UgPSBmYWxzZTtcbkNvZGVNaXJyb3IuY29tbWFuZHMuc2Nyb2xsU2VsZWN0aW9uVG9DZW50ZXIgPSBmdW5jdGlvbiAoY20pIHtcbiAgICB2YXIgY3Vyc29yID0gY20uZ2V0Q3Vyc29yKCdoZWFkJyk7XG4gICAgdmFyIGNoYXJDb29yZHMgPSBjbS5jaGFyQ29vcmRzKGN1cnNvciwgXCJsb2NhbFwiKTtcbiAgICB2YXIgdG9wID0gY2hhckNvb3Jkcy50b3A7XG4gICAgdmFyIGhhbGZMaW5lSGVpZ2h0ID0gKGNoYXJDb29yZHMuYm90dG9tIC0gdG9wKSAvIDI7XG4gICAgdmFyIGhhbGZXaW5kb3dIZWlnaHQgPSBjbS5nZXRXcmFwcGVyRWxlbWVudCgpLm9mZnNldEhlaWdodCAvIDI7XG4gICAgdmFyIHNjcm9sbFRvID0gTWF0aC5yb3VuZCgodG9wIC0gaGFsZldpbmRvd0hlaWdodCArIGhhbGZMaW5lSGVpZ2h0KSk7XG4gICAgY20uc2Nyb2xsVG8obnVsbCwgc2Nyb2xsVG8pO1xufTtcbkNvZGVNaXJyb3IuZGVmaW5lT3B0aW9uKFwidHlwZXdyaXRlclNjcm9sbGluZ1wiLCBmYWxzZSwgZnVuY3Rpb24gKGNtLCB2YWwsIG9sZCkge1xuICAgIGlmIChvbGQgJiYgb2xkICE9IENvZGVNaXJyb3IuSW5pdCkge1xuICAgICAgICBjb25zdCBsaW5lc0VsID0gY20uZ2V0U2Nyb2xsZXJFbGVtZW50KCkucXVlcnlTZWxlY3RvcignLkNvZGVNaXJyb3ItbGluZXMnKTtcbiAgICAgICAgbGluZXNFbC5zdHlsZS5wYWRkaW5nVG9wID0gbnVsbDtcbiAgICAgICAgbGluZXNFbC5zdHlsZS5wYWRkaW5nQm90dG9tID0gbnVsbDtcbiAgICAgICAgY20ub2ZmKFwiY3Vyc29yQWN0aXZpdHlcIiwgb25DdXJzb3JBY3Rpdml0eSk7XG4gICAgICAgIGNtLm9mZihcInJlZnJlc2hcIiwgb25SZWZyZXNoKTtcbiAgICAgICAgY20ub2ZmKFwibW91c2Vkb3duXCIsIG9uTW91c2VEb3duKTtcbiAgICAgICAgY20ub2ZmKFwia2V5ZG93blwiLCBvbktleURvd24pO1xuICAgICAgICBjbS5vZmYoXCJiZWZvcmVDaGFuZ2VcIiwgb25CZWZvcmVDaGFuZ2UpO1xuICAgIH1cbiAgICBpZiAodmFsKSB7XG4gICAgICAgIG9uUmVmcmVzaChjbSk7XG4gICAgICAgIGNtLm9uKFwiY3Vyc29yQWN0aXZpdHlcIiwgb25DdXJzb3JBY3Rpdml0eSk7XG4gICAgICAgIGNtLm9uKFwicmVmcmVzaFwiLCBvblJlZnJlc2gpO1xuICAgICAgICBjbS5vbihcIm1vdXNlZG93blwiLCBvbk1vdXNlRG93bik7XG4gICAgICAgIGNtLm9uKFwia2V5ZG93blwiLCBvbktleURvd24pO1xuICAgICAgICBjbS5vbihcImJlZm9yZUNoYW5nZVwiLCBvbkJlZm9yZUNoYW5nZSk7XG4gICAgfVxufSk7XG5mdW5jdGlvbiBvbk1vdXNlRG93bigpIHtcbiAgICBtb3ZlZEJ5TW91c2UgPSB0cnVlO1xufVxuY29uc3QgbW9kaWZlcktleXMgPSBbXCJBbHRcIiwgXCJBbHRHcmFwaFwiLCBcIkNhcHNMb2NrXCIsIFwiQ29udHJvbFwiLCBcIkZuXCIsIFwiRm5Mb2NrXCIsIFwiSHlwZXJcIiwgXCJNZXRhXCIsIFwiTnVtTG9ja1wiLCBcIlNjcm9sbExvY2tcIiwgXCJTaGlmdFwiLCBcIlN1cGVyXCIsIFwiU3ltYm9sXCIsIFwiU3ltYm9sTG9ja1wiXTtcbmZ1bmN0aW9uIG9uS2V5RG93bihjbSwgZSkge1xuICAgIGlmICghbW9kaWZlcktleXMuaW5jbHVkZXMoZS5rZXkpKSB7XG4gICAgICAgIG1vdmVkQnlNb3VzZSA9IGZhbHNlO1xuICAgIH1cbn1cbmZ1bmN0aW9uIG9uQmVmb3JlQ2hhbmdlKCkge1xuICAgIG1vdmVkQnlNb3VzZSA9IGZhbHNlO1xufVxuZnVuY3Rpb24gb25DdXJzb3JBY3Rpdml0eShjbSkge1xuICAgIGNvbnN0IGxpbmVzRWwgPSBjbS5nZXRTY3JvbGxlckVsZW1lbnQoKS5xdWVyeVNlbGVjdG9yKCcuQ29kZU1pcnJvci1saW5lcycpO1xuICAgIGlmIChjbS5nZXRTZWxlY3Rpb24oKS5sZW5ndGggIT09IDApIHtcbiAgICAgICAgbGluZXNFbC5jbGFzc0xpc3QuYWRkKFwic2VsZWN0aW5nXCIpO1xuICAgIH1cbiAgICBlbHNlIHtcbiAgICAgICAgbGluZXNFbC5jbGFzc0xpc3QucmVtb3ZlKFwic2VsZWN0aW5nXCIpO1xuICAgIH1cblxuICAgIGlmKCFtb3ZlZEJ5TW91c2UpIHtcbiAgICAgICAgY20uZXhlY0NvbW1hbmQoXCJzY3JvbGxTZWxlY3Rpb25Ub0NlbnRlclwiKTtcbiAgICB9XG59XG5mdW5jdGlvbiBvblJlZnJlc2goY20pIHtcbiAgICBjb25zdCBoYWxmV2luZG93SGVpZ2h0ID0gY20uZ2V0V3JhcHBlckVsZW1lbnQoKS5vZmZzZXRIZWlnaHQgLyAyO1xuICAgIGNvbnN0IGxpbmVzRWwgPSBjbS5nZXRTY3JvbGxlckVsZW1lbnQoKS5xdWVyeVNlbGVjdG9yKCcuQ29kZU1pcnJvci1saW5lcycpO1xuICAgIGxpbmVzRWwuc3R5bGUucGFkZGluZ1RvcCA9IGAke2hhbGZXaW5kb3dIZWlnaHR9cHhgO1xuICAgIGxpbmVzRWwuc3R5bGUucGFkZGluZ0JvdHRvbSA9IGAke2hhbGZXaW5kb3dIZWlnaHR9cHhgOyAvLyBUaGFua3MgQHdhbHVsdWxhIVxuICAgIGlmIChjbS5nZXRTZWxlY3Rpb24oKS5sZW5ndGggPT09IDApIHtcbiAgICAgICAgY20uZXhlY0NvbW1hbmQoXCJzY3JvbGxTZWxlY3Rpb25Ub0NlbnRlclwiKTtcbiAgICB9XG59XG4iLCJpbXBvcnQgeyBFeHRlbnNpb24sIFNlbGVjdGlvblJhbmdlLCBFZGl0b3JTdGF0ZSwgRWRpdG9yU2VsZWN0aW9uLCBUcmFuc2FjdGlvbiwgUHJlYyB9IGZyb20gXCJAY29kZW1pcnJvci9zdGF0ZVwiXG5pbXBvcnQgeyBWaWV3UGx1Z2luLCBWaWV3VXBkYXRlLCBFZGl0b3JWaWV3IH0gZnJvbSBcIkBjb2RlbWlycm9yL3ZpZXdcIlxuZGVjbGFyZSB0eXBlIFNjcm9sbFN0cmF0ZWd5ID0gXCJuZWFyZXN0XCIgfCBcInN0YXJ0XCIgfCBcImVuZFwiIHwgXCJjZW50ZXJcIjtcblxuY29uc3QgYWxsb3dlZFVzZXJFdmVudHMgPSAvXihzZWxlY3R8aW5wdXR8ZGVsZXRlfHVuZG98cmVkbykoXFwuLispPyQvXG5jb25zdCBkaXNhbGxvd2VkVXNlckV2ZW50cyA9IC9eKHNlbGVjdC5wb2ludGVyKSQvXG5cbmNvbnN0IHR5cGV3cml0ZXJTY3JvbGxQbHVnaW4gPSBWaWV3UGx1Z2luLmZyb21DbGFzcyhjbGFzcyB7XG4gIHByaXZhdGUgbXlVcGRhdGUgPSBmYWxzZTtcbiAgcHJpdmF0ZSBwYWRkaW5nOnN0cmluZyA9IG51bGw7XG4gIFxuICBjb25zdHJ1Y3Rvcihwcml2YXRlIHZpZXc6IEVkaXRvclZpZXcpIHsgfVxuXG4gIHVwZGF0ZSh1cGRhdGU6IFZpZXdVcGRhdGUpIHtcbiAgICBpZiAodGhpcy5teVVwZGF0ZSkgdGhpcy5teVVwZGF0ZSA9IGZhbHNlO1xuICAgIGVsc2Uge1xuICAgICAgdGhpcy5wYWRkaW5nID0gKCh0aGlzLnZpZXcuZG9tLmNsaWVudEhlaWdodCAvIDIpIC0gKHRoaXMudmlldy5kZWZhdWx0TGluZUhlaWdodCkpICsgXCJweFwiXG4gICAgICBpZiAodGhpcy5wYWRkaW5nICE9IHRoaXMudmlldy5jb250ZW50RE9NLnN0eWxlLnBhZGRpbmdUb3ApIHtcbiAgICAgICAgdGhpcy52aWV3LmNvbnRlbnRET00uc3R5bGUucGFkZGluZ1RvcCA9IHRoaXMudmlldy5jb250ZW50RE9NLnN0eWxlLnBhZGRpbmdCb3R0b20gPSB0aGlzLnBhZGRpbmdcbiAgICAgIH1cblxuICAgICAgY29uc3QgdXNlckV2ZW50cyA9IHVwZGF0ZS50cmFuc2FjdGlvbnMubWFwKHRyID0+IHRyLmFubm90YXRpb24oVHJhbnNhY3Rpb24udXNlckV2ZW50KSlcbiAgICAgIGNvbnN0IGlzQWxsb3dlZCA9IHVzZXJFdmVudHMucmVkdWNlPGJvb2xlYW4+KFxuICAgICAgICAocmVzdWx0LCBldmVudCkgPT4gcmVzdWx0ICYmIGFsbG93ZWRVc2VyRXZlbnRzLnRlc3QoZXZlbnQpICYmICFkaXNhbGxvd2VkVXNlckV2ZW50cy50ZXN0KGV2ZW50KSxcbiAgICAgICAgdXNlckV2ZW50cy5sZW5ndGggPiAwXG4gICAgICApO1xuICAgICAgaWYgKGlzQWxsb3dlZCkge1xuICAgICAgICB0aGlzLm15VXBkYXRlID0gdHJ1ZTtcbiAgICAgICAgdGhpcy5jZW50ZXJPbkhlYWQodXBkYXRlKVxuICAgICAgfVxuICAgIH1cbiAgfVxuXG4gIGFzeW5jIGNlbnRlck9uSGVhZCh1cGRhdGU6IFZpZXdVcGRhdGUpIHtcbiAgICAvLyBjYW4ndCB1cGRhdGUgaW5zaWRlIGFuIHVwZGF0ZSwgc28gcmVxdWVzdCB0aGUgbmV4dCBhbmltYXRpb24gZnJhbWVcbiAgICB3aW5kb3cucmVxdWVzdEFuaW1hdGlvbkZyYW1lKCgpID0+IHtcbiAgICAgIC8vIGN1cnJlbnQgc2VsZWN0aW9uIHJhbmdlXG4gICAgICBpZiAodXBkYXRlLnN0YXRlLnNlbGVjdGlvbi5yYW5nZXMubGVuZ3RoID09IDEpIHtcbiAgICAgICAgLy8gb25seSBhY3Qgb24gdGhlIG9uZSByYW5nZVxuICAgICAgICBjb25zdCBoZWFkID0gdXBkYXRlLnN0YXRlLnNlbGVjdGlvbi5tYWluLmhlYWQ7XG4gICAgICAgIGNvbnN0IHByZXZIZWFkID0gdXBkYXRlLnN0YXJ0U3RhdGUuc2VsZWN0aW9uLm1haW4uaGVhZDtcbiAgICAgICAgLy8gVE9ETzogd29yayBvdXQgbGluZSBudW1iZXIgYW5kIHVzZSB0aGF0IGluc3RlYWQ/IElzIHRoYXQgZXZlbiBwb3NzaWJsZT9cbiAgICAgICAgLy8gZG9uJ3QgYm90aGVyIHdpdGggdGhpcyBuZXh0IHBhcnQgaWYgdGhlIHJhbmdlIChsaW5lPz8pIGhhc24ndCBjaGFuZ2VkXG4gICAgICAgIGlmIChwcmV2SGVhZCAhPSBoZWFkKSB7XG4gICAgICAgICAgLy8gdGhpcyBpcyB0aGUgZWZmZWN0IHRoYXQgZG9lcyB0aGUgY2VudGVyaW5nXG4gICAgICAgICAgY29uc3QgZWZmZWN0ID0gRWRpdG9yVmlldy5zY3JvbGxJbnRvVmlldyhoZWFkLCB7IHk6IFwiY2VudGVyXCIgfSlcbiAgICAgICAgICBjb25zdCB0cmFuc2FjdGlvbiA9IHRoaXMudmlldy5zdGF0ZS51cGRhdGUoeyBlZmZlY3RzOiBlZmZlY3QgfSk7XG4gICAgICAgICAgdGhpcy52aWV3LmRpc3BhdGNoKHRyYW5zYWN0aW9uKVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfSlcbiAgfVxufSlcblxuZXhwb3J0IGZ1bmN0aW9uIHR5cGV3cml0ZXJTY3JvbGwob3B0aW9uczogYW55ID0ge30pOiBFeHRlbnNpb24ge1xuICByZXR1cm4gW1xuICAgIHR5cGV3cml0ZXJTY3JvbGxQbHVnaW5cbiAgXVxufSIsImltcG9ydCAnLi9zdHlsZXMuc2NzcydcbmltcG9ydCAnLi90eXBld3JpdGVyLXNjcm9sbGluZydcbmltcG9ydCB7IFBsdWdpbiwgTWFya2Rvd25WaWV3LCBQbHVnaW5TZXR0aW5nVGFiLCBBcHAsIFNldHRpbmcgfSBmcm9tICdvYnNpZGlhbidcbmltcG9ydCB7IHR5cGV3cml0ZXJTY3JvbGwgfSBmcm9tICcuL2V4dGVuc2lvbidcblxuY2xhc3MgQ01UeXBld3JpdGVyU2Nyb2xsU2V0dGluZ3Mge1xuICBlbmFibGVkOiBib29sZWFuO1xuICB6ZW5FbmFibGVkOiBib29sZWFuO1xufVxuXG5jb25zdCBERUZBVUxUX1NFVFRJTkdTOiBDTVR5cGV3cml0ZXJTY3JvbGxTZXR0aW5ncyA9IHtcbiAgZW5hYmxlZDogdHJ1ZSxcbiAgemVuRW5hYmxlZDogZmFsc2Vcbn1cblxuZXhwb3J0IGRlZmF1bHQgY2xhc3MgQ01UeXBld3JpdGVyU2Nyb2xsUGx1Z2luIGV4dGVuZHMgUGx1Z2luIHtcbiAgc2V0dGluZ3M6IENNVHlwZXdyaXRlclNjcm9sbFNldHRpbmdzO1xuXG4gIGFzeW5jIG9ubG9hZCgpIHtcbiAgICB0aGlzLnNldHRpbmdzID0gT2JqZWN0LmFzc2lnbihERUZBVUxUX1NFVFRJTkdTLCBhd2FpdCB0aGlzLmxvYWREYXRhKCkpO1xuICAgIFxuICAgIC8vIGVuYWJsZSB0aGUgcGx1Z2luIChiYXNlZCBvbiBzZXR0aW5ncylcbiAgICBpZiAodGhpcy5zZXR0aW5ncy5lbmFibGVkKSB0aGlzLmVuYWJsZVR5cGV3cml0ZXJTY3JvbGwoKTtcbiAgICBpZiAodGhpcy5zZXR0aW5ncy56ZW5FbmFibGVkKSB0aGlzLmVuYWJsZVplbigpO1xuXG4gICAgLy8gYWRkIHRoZSBzZXR0aW5ncyB0YWJcbiAgICB0aGlzLmFkZFNldHRpbmdUYWIobmV3IENNVHlwZXdyaXRlclNjcm9sbFNldHRpbmdUYWIodGhpcy5hcHAsIHRoaXMpKTtcblxuICAgIC8vIGFkZCB0aGUgY29tbWFuZHMgLyBrZXlib2FyZCBzaG9ydGN1dHNcbiAgICB0aGlzLmFkZENvbW1hbmRzKCk7XG4gIH1cblxuICBvbnVubG9hZCgpIHtcbiAgICAvLyBkaXNhYmxlIHRoZSBwbHVnaW5cbiAgICB0aGlzLmRpc2FibGVUeXBld3JpdGVyU2Nyb2xsKCk7XG4gICAgdGhpcy5kaXNhYmxlWmVuKCk7XG4gIH1cblxuICBhZGRDb21tYW5kcygpIHsgXG4gICAgLy8gYWRkIHRoZSB0b2dnbGUgb24vb2ZmIGNvbW1hbmRcbiAgICB0aGlzLmFkZENvbW1hbmQoe1xuICAgICAgaWQ6ICd0b2dnbGUtdHlwZXdyaXRlci1zcm9sbCcsXG4gICAgICBuYW1lOiAnVG9nZ2xlIE9uL09mZicsXG4gICAgICBjYWxsYmFjazogKCkgPT4geyB0aGlzLnRvZ2dsZVR5cGV3cml0ZXJTY3JvbGwoKTsgfVxuICAgIH0pO1xuXG4gICAgLy8gdG9nZ2xlIHplbiBtb2RlXG4gICAgdGhpcy5hZGRDb21tYW5kKHtcbiAgICAgIGlkOiAndG9nZ2xlLXR5cGV3cml0ZXItc3JvbGwtemVuJyxcbiAgICAgIG5hbWU6ICdUb2dnbGUgWmVuIE1vZGUgT24vT2ZmJyxcbiAgICAgIGNhbGxiYWNrOiAoKSA9PiB7IHRoaXMudG9nZ2xlWmVuKCk7IH1cbiAgICB9KTtcbiAgfVxuXG4gIHRvZ2dsZVR5cGV3cml0ZXJTY3JvbGwgPSAobmV3VmFsdWU6IGJvb2xlYW4gPSBudWxsKSA9PiB7XG4gICAgLy8gaWYgbm8gdmFsdWUgaXMgcGFzc2VkIGluLCB0b2dnbGUgdGhlIGV4aXN0aW5nIHZhbHVlXG4gICAgaWYgKG5ld1ZhbHVlID09PSBudWxsKSBuZXdWYWx1ZSA9ICF0aGlzLnNldHRpbmdzLmVuYWJsZWQ7XG4gICAgLy8gYXNzaWduIHRoZSBuZXcgdmFsdWUgYW5kIGNhbGwgdGhlIGNvcnJlY3QgZW5hYmxlIC8gZGlzYWJsZSBmdW5jdGlvblxuICAgICh0aGlzLnNldHRpbmdzLmVuYWJsZWQgPSBuZXdWYWx1ZSlcbiAgICAgID8gdGhpcy5lbmFibGVUeXBld3JpdGVyU2Nyb2xsKCkgOiB0aGlzLmRpc2FibGVUeXBld3JpdGVyU2Nyb2xsKCk7XG4gICAgLy8gc2F2ZSB0aGUgbmV3IHNldHRpbmdzXG4gICAgdGhpcy5zYXZlRGF0YSh0aGlzLnNldHRpbmdzKTtcbiAgfVxuXG4gIHRvZ2dsZVplbiA9IChuZXdWYWx1ZTogYm9vbGVhbiA9IG51bGwpID0+IHtcbiAgICAvLyBpZiBubyB2YWx1ZSBpcyBwYXNzZWQgaW4sIHRvZ2dsZSB0aGUgZXhpc3RpbmcgdmFsdWVcbiAgICBpZiAobmV3VmFsdWUgPT09IG51bGwpIG5ld1ZhbHVlID0gIXRoaXMuc2V0dGluZ3MuemVuRW5hYmxlZDtcbiAgICAvLyBhc3NpZ24gdGhlIG5ldyB2YWx1ZSBhbmQgY2FsbCB0aGUgY29ycmVjdCBlbmFibGUgLyBkaXNhYmxlIGZ1bmN0aW9uXG4gICAgKHRoaXMuc2V0dGluZ3MuemVuRW5hYmxlZCA9IG5ld1ZhbHVlKVxuICAgICAgPyB0aGlzLmVuYWJsZVplbigpIDogdGhpcy5kaXNhYmxlWmVuKCk7XG4gICAgLy8gc2F2ZSB0aGUgbmV3IHNldHRpbmdzXG4gICAgdGhpcy5zYXZlRGF0YSh0aGlzLnNldHRpbmdzKTtcbiAgfVxuXG4gIGVuYWJsZVR5cGV3cml0ZXJTY3JvbGwgPSAoKSA9PiB7XG4gICAgLy8gYWRkIHRoZSBjbGFzc1xuICAgIGRvY3VtZW50LmJvZHkuY2xhc3NMaXN0LmFkZCgncGx1Z2luLWNtLXR5cGV3cml0ZXItc2Nyb2xsJyk7XG5cbiAgICAvLyByZWdpc3RlciB0aGUgY29kZW1pcnJvciBhZGQgb24gc2V0dGluZ1xuICAgIHRoaXMucmVnaXN0ZXJDb2RlTWlycm9yKGNtID0+IHtcbiAgICAgIC8vIEB0cy1pZ25vcmVcbiAgICAgIGNtLnNldE9wdGlvbihcInR5cGV3cml0ZXJTY3JvbGxpbmdcIiwgdHJ1ZSk7XG4gICAgfSk7XG5cbiAgICB0aGlzLnJlZ2lzdGVyRWRpdG9yRXh0ZW5zaW9uKHR5cGV3cml0ZXJTY3JvbGwoKSlcbiAgfVxuICBcbiAgZGlzYWJsZVR5cGV3cml0ZXJTY3JvbGwgPSAoKSA9PiB7XG4gICAgLy8gcmVtb3ZlIHRoZSBjbGFzc1xuICAgIGRvY3VtZW50LmJvZHkuY2xhc3NMaXN0LnJlbW92ZSgncGx1Z2luLWNtLXR5cGV3cml0ZXItc2Nyb2xsJyk7XG5cbiAgICAvLyByZW1vdmUgdGhlIGNvZGVtaXJyb3IgYWRkIG9uIHNldHRpbmdcbiAgICB0aGlzLmFwcC53b3Jrc3BhY2UuaXRlcmF0ZUNvZGVNaXJyb3JzKGNtID0+IHtcbiAgICAgIC8vIEB0cy1pZ25vcmVcbiAgICAgIGNtLnNldE9wdGlvbihcInR5cGV3cml0ZXJTY3JvbGxpbmdcIiwgZmFsc2UpO1xuICAgIH0pO1xuICB9XG5cbiAgZW5hYmxlWmVuID0gKCkgPT4ge1xuICAgIC8vIGFkZCB0aGUgY2xhc3NcbiAgICBkb2N1bWVudC5ib2R5LmNsYXNzTGlzdC5hZGQoJ3BsdWdpbi1jbS10eXBld3JpdGVyLXNjcm9sbC16ZW4nKTtcbiAgfVxuXG4gIGRpc2FibGVaZW4gPSAoKSA9PiB7XG4gICAgLy8gcmVtb3ZlIHRoZSBjbGFzc1xuICAgIGRvY3VtZW50LmJvZHkuY2xhc3NMaXN0LnJlbW92ZSgncGx1Z2luLWNtLXR5cGV3cml0ZXItc2Nyb2xsLXplbicpO1xuICB9XG59XG5cbmNsYXNzIENNVHlwZXdyaXRlclNjcm9sbFNldHRpbmdUYWIgZXh0ZW5kcyBQbHVnaW5TZXR0aW5nVGFiIHtcblxuICBwbHVnaW46IENNVHlwZXdyaXRlclNjcm9sbFBsdWdpbjtcbiAgY29uc3RydWN0b3IoYXBwOiBBcHAsIHBsdWdpbjogQ01UeXBld3JpdGVyU2Nyb2xsUGx1Z2luKSB7XG4gICAgc3VwZXIoYXBwLCBwbHVnaW4pO1xuICAgIHRoaXMucGx1Z2luID0gcGx1Z2luO1xuICB9XG5cbiAgZGlzcGxheSgpOiB2b2lkIHtcbiAgICBsZXQgeyBjb250YWluZXJFbCB9ID0gdGhpcztcblxuICAgIGNvbnRhaW5lckVsLmVtcHR5KCk7XG5cbiAgICBuZXcgU2V0dGluZyhjb250YWluZXJFbClcbiAgICAgIC5zZXROYW1lKFwiVG9nZ2xlIFR5cGV3cml0ZXIgU2Nyb2xsaW5nXCIpXG4gICAgICAuc2V0RGVzYyhcIlR1cm5zIHR5cGV3cml0ZXIgc2Nyb2xsaW5nIG9uIG9yIG9mZiBnbG9iYWxseVwiKVxuICAgICAgLmFkZFRvZ2dsZSh0b2dnbGUgPT5cbiAgICAgICAgdG9nZ2xlLnNldFZhbHVlKHRoaXMucGx1Z2luLnNldHRpbmdzLmVuYWJsZWQpXG4gICAgICAgICAgLm9uQ2hhbmdlKChuZXdWYWx1ZSkgPT4geyB0aGlzLnBsdWdpbi50b2dnbGVUeXBld3JpdGVyU2Nyb2xsKG5ld1ZhbHVlKSB9KVxuICAgICAgKTtcblxuICAgIG5ldyBTZXR0aW5nKGNvbnRhaW5lckVsKVxuICAgICAgLnNldE5hbWUoXCJaZW4gTW9kZVwiKVxuICAgICAgLnNldERlc2MoXCJEYXJrZW5zIG5vbi1hY3RpdmUgbGluZXMgaW4gdGhlIGVkaXRvciBzbyB5b3UgY2FuIGZvY3VzIG9uIHdoYXQgeW91J3JlIHR5cGluZ1wiKVxuICAgICAgLmFkZFRvZ2dsZSh0b2dnbGUgPT5cbiAgICAgICAgdG9nZ2xlLnNldFZhbHVlKHRoaXMucGx1Z2luLnNldHRpbmdzLnplbkVuYWJsZWQpXG4gICAgICAgICAgLm9uQ2hhbmdlKChuZXdWYWx1ZSkgPT4geyB0aGlzLnBsdWdpbi50b2dnbGVaZW4obmV3VmFsdWUpIH0pXG4gICAgICApO1xuXG4gIH1cbn1cbiJdLCJuYW1lcyI6WyJWaWV3UGx1Z2luIiwiVHJhbnNhY3Rpb24iLCJFZGl0b3JWaWV3IiwiUGx1Z2luIiwiU2V0dGluZyIsIlBsdWdpblNldHRpbmdUYWIiXSwibWFwcGluZ3MiOiI7Ozs7OztBQUFBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsSUFBSSxhQUFhLEdBQUcsU0FBUyxDQUFDLEVBQUUsQ0FBQyxFQUFFO0FBQ25DLElBQUksYUFBYSxHQUFHLE1BQU0sQ0FBQyxjQUFjO0FBQ3pDLFNBQVMsRUFBRSxTQUFTLEVBQUUsRUFBRSxFQUFFLFlBQVksS0FBSyxJQUFJLFVBQVUsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUMsQ0FBQyxTQUFTLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQztBQUNwRixRQUFRLFVBQVUsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFLEtBQUssSUFBSSxDQUFDLElBQUksQ0FBQyxFQUFFLElBQUksTUFBTSxDQUFDLFNBQVMsQ0FBQyxjQUFjLENBQUMsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQztBQUMxRyxJQUFJLE9BQU8sYUFBYSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztBQUMvQixDQUFDLENBQUM7QUFDRjtBQUNPLFNBQVMsU0FBUyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUU7QUFDaEMsSUFBSSxJQUFJLE9BQU8sQ0FBQyxLQUFLLFVBQVUsSUFBSSxDQUFDLEtBQUssSUFBSTtBQUM3QyxRQUFRLE1BQU0sSUFBSSxTQUFTLENBQUMsc0JBQXNCLEdBQUcsTUFBTSxDQUFDLENBQUMsQ0FBQyxHQUFHLCtCQUErQixDQUFDLENBQUM7QUFDbEcsSUFBSSxhQUFhLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0FBQ3hCLElBQUksU0FBUyxFQUFFLEdBQUcsRUFBRSxJQUFJLENBQUMsV0FBVyxHQUFHLENBQUMsQ0FBQyxFQUFFO0FBQzNDLElBQUksQ0FBQyxDQUFDLFNBQVMsR0FBRyxDQUFDLEtBQUssSUFBSSxHQUFHLE1BQU0sQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLElBQUksRUFBRSxDQUFDLFNBQVMsR0FBRyxDQUFDLENBQUMsU0FBUyxFQUFFLElBQUksRUFBRSxFQUFFLENBQUMsQ0FBQztBQUN6RixDQUFDO0FBdUNEO0FBQ08sU0FBUyxTQUFTLENBQUMsT0FBTyxFQUFFLFVBQVUsRUFBRSxDQUFDLEVBQUUsU0FBUyxFQUFFO0FBQzdELElBQUksU0FBUyxLQUFLLENBQUMsS0FBSyxFQUFFLEVBQUUsT0FBTyxLQUFLLFlBQVksQ0FBQyxHQUFHLEtBQUssR0FBRyxJQUFJLENBQUMsQ0FBQyxVQUFVLE9BQU8sRUFBRSxFQUFFLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFO0FBQ2hILElBQUksT0FBTyxLQUFLLENBQUMsS0FBSyxDQUFDLEdBQUcsT0FBTyxDQUFDLEVBQUUsVUFBVSxPQUFPLEVBQUUsTUFBTSxFQUFFO0FBQy9ELFFBQVEsU0FBUyxTQUFTLENBQUMsS0FBSyxFQUFFLEVBQUUsSUFBSSxFQUFFLElBQUksQ0FBQyxTQUFTLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLE9BQU8sQ0FBQyxFQUFFLEVBQUUsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRTtBQUNuRyxRQUFRLFNBQVMsUUFBUSxDQUFDLEtBQUssRUFBRSxFQUFFLElBQUksRUFBRSxJQUFJLENBQUMsU0FBUyxDQUFDLE9BQU8sQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLE9BQU8sQ0FBQyxFQUFFLEVBQUUsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRTtBQUN0RyxRQUFRLFNBQVMsSUFBSSxDQUFDLE1BQU0sRUFBRSxFQUFFLE1BQU0sQ0FBQyxJQUFJLEdBQUcsT0FBTyxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsR0FBRyxLQUFLLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLElBQUksQ0FBQyxTQUFTLEVBQUUsUUFBUSxDQUFDLENBQUMsRUFBRTtBQUN0SCxRQUFRLElBQUksQ0FBQyxDQUFDLFNBQVMsR0FBRyxTQUFTLENBQUMsS0FBSyxDQUFDLE9BQU8sRUFBRSxVQUFVLElBQUksRUFBRSxDQUFDLEVBQUUsSUFBSSxFQUFFLENBQUMsQ0FBQztBQUM5RSxLQUFLLENBQUMsQ0FBQztBQUNQLENBQUM7QUFDRDtBQUNPLFNBQVMsV0FBVyxDQUFDLE9BQU8sRUFBRSxJQUFJLEVBQUU7QUFDM0MsSUFBSSxJQUFJLENBQUMsR0FBRyxFQUFFLEtBQUssRUFBRSxDQUFDLEVBQUUsSUFBSSxFQUFFLFdBQVcsRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLEVBQUUsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLEVBQUUsSUFBSSxFQUFFLEVBQUUsRUFBRSxHQUFHLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDO0FBQ3JILElBQUksT0FBTyxDQUFDLEdBQUcsRUFBRSxJQUFJLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLE9BQU8sRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsUUFBUSxFQUFFLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxFQUFFLE9BQU8sTUFBTSxLQUFLLFVBQVUsS0FBSyxDQUFDLENBQUMsTUFBTSxDQUFDLFFBQVEsQ0FBQyxHQUFHLFdBQVcsRUFBRSxPQUFPLElBQUksQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUM7QUFDN0osSUFBSSxTQUFTLElBQUksQ0FBQyxDQUFDLEVBQUUsRUFBRSxPQUFPLFVBQVUsQ0FBQyxFQUFFLEVBQUUsT0FBTyxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRTtBQUN0RSxJQUFJLFNBQVMsSUFBSSxDQUFDLEVBQUUsRUFBRTtBQUN0QixRQUFRLElBQUksQ0FBQyxFQUFFLE1BQU0sSUFBSSxTQUFTLENBQUMsaUNBQWlDLENBQUMsQ0FBQztBQUN0RSxRQUFRLE9BQU8sQ0FBQyxFQUFFLElBQUk7QUFDdEIsWUFBWSxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsQ0FBQyxRQUFRLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxRQUFRLENBQUMsS0FBSyxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLElBQUksRUFBRSxPQUFPLENBQUMsQ0FBQztBQUN6SyxZQUFZLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUM7QUFDcEQsWUFBWSxRQUFRLEVBQUUsQ0FBQyxDQUFDLENBQUM7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
