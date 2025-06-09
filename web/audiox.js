// AudioX ComfyUI Web Components
// This file contains web-side functionality for AudioX nodes

import { app } from "../../scripts/app.js";

// Extension registration with post-execution patching
app.registerExtension({
    name: "AudioX.WebComponents",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Add beforeQueued method if it doesn't exist
        if (!nodeType.prototype.beforeQueued) {
            nodeType.prototype.beforeQueued = function() {
                console.log("AudioX: beforeQueued called on", nodeData.name);
                return true;
            };
        }
    },

    async nodeCreated(node) {
        // Ensure node has beforeQueued method
        if (!node.beforeQueued) {
            node.beforeQueued = function() {
                console.log("AudioX: beforeQueued called on node", node.type);
                return true;
            };
        }
    },

    // CRITICAL: Re-patch after workflow execution
    async beforeQueued(details) {
        console.log("AudioX: Extension beforeQueued called");

        // Re-patch all nodes after execution to prevent second-run errors
        setTimeout(() => {
            this.repatchAllNodes();
        }, 100);

        return true;
    },

    // Re-patch nodes after execution
    repatchAllNodes() {
        try {
            console.log("AudioX: Re-patching all nodes after execution...");

            if (window.app?.graph?._nodes) {
                window.app.graph._nodes.forEach((node, index) => {
                    if (node && (!node.beforeQueued || typeof node.beforeQueued !== 'function')) {
                        node.beforeQueued = function() {
                            console.log(`AudioX: Re-patched beforeQueued called on node ${index} (${node.type})`);
                            return true;
                        };
                        console.log(`AudioX: Re-patched node ${index}: ${node.type}`);
                    }
                });
            }

            // Also re-patch prototypes in case they got reset
            if (window.LiteGraph?.LGraphNode?.prototype && !window.LiteGraph.LGraphNode.prototype.beforeQueued) {
                window.LiteGraph.LGraphNode.prototype.beforeQueued = function() {
                    console.log("AudioX: Re-patched LiteGraph beforeQueued called");
                    return true;
                };
                console.log("AudioX: Re-patched LiteGraph prototype");
            }

        } catch (error) {
            console.warn("AudioX: Error re-patching nodes:", error);
        }
    }
});

// Monitor for workflow execution completion and re-patch
function setupPostExecutionMonitoring() {
    try {
        // Monitor for execution completion via app events
        if (window.app) {
            // Hook into the app's execution system
            const originalQueuePrompt = window.app.queuePrompt;
            if (originalQueuePrompt) {
                window.app.queuePrompt = async function(...args) {
                    console.log("AudioX: Workflow execution starting...");

                    try {
                        const result = await originalQueuePrompt.apply(this, args);

                        // Re-patch after execution completes
                        setTimeout(() => {
                            console.log("AudioX: Post-execution re-patching...");
                            repatchAfterExecution();
                        }, 500);

                        return result;
                    } catch (error) {
                        console.warn("AudioX: Error in queuePrompt:", error);
                        // Still re-patch even if execution failed
                        setTimeout(() => {
                            repatchAfterExecution();
                        }, 500);
                        throw error;
                    }
                };
                console.log("AudioX: Hooked into queuePrompt for post-execution patching");
            }
        }

        // Also monitor for WebSocket messages that indicate execution completion
        if (window.WebSocket) {
            const originalWebSocket = window.WebSocket;
            window.WebSocket = function(...args) {
                const ws = new originalWebSocket(...args);

                const originalOnMessage = ws.onmessage;
                ws.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);

                        // Check for execution completion messages
                        if (data.type === 'executed' || data.type === 'execution_cached') {
                            console.log("AudioX: Detected execution completion via WebSocket");
                            setTimeout(() => {
                                repatchAfterExecution();
                            }, 200);
                        }
                    } catch (e) {
                        // Ignore JSON parse errors
                    }

                    if (originalOnMessage) {
                        return originalOnMessage.apply(this, arguments);
                    }
                };

                return ws;
            };
            console.log("AudioX: Set up WebSocket monitoring for execution completion");
        }

    } catch (error) {
        console.warn("AudioX: Error setting up post-execution monitoring:", error);
    }
}

// Function to re-patch everything after execution
function repatchAfterExecution() {
    try {
        console.log("AudioX: Re-patching after execution...");

        // Re-patch all nodes
        if (window.app?.graph?._nodes) {
            window.app.graph._nodes.forEach((node, index) => {
                if (node && (!node.beforeQueued || typeof node.beforeQueued !== 'function')) {
                    node.beforeQueued = function() {
                        console.log(`AudioX: Post-exec beforeQueued on node ${index} (${node.type})`);
                        return true;
                    };
                }
            });
        }

        // Re-patch prototypes
        if (window.LiteGraph?.LGraphNode?.prototype && !window.LiteGraph.LGraphNode.prototype.beforeQueued) {
            window.LiteGraph.LGraphNode.prototype.beforeQueued = function() {
                console.log("AudioX: Post-exec LiteGraph beforeQueued");
                return true;
            };
        }

        // Re-patch app object
        if (window.app && !window.app.beforeQueued) {
            window.app.beforeQueued = function() {
                console.log("AudioX: Post-exec app beforeQueued");
                return true;
            };
        }

        console.log("AudioX: Post-execution re-patching completed");

    } catch (error) {
        console.warn("AudioX: Error in post-execution re-patching:", error);
    }
}

// Set up monitoring after a short delay
setTimeout(setupPostExecutionMonitoring, 1000);

console.log("AudioX: Extension with post-execution monitoring loaded");
