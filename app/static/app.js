(function() {
    // State management
    let activeDocumentId = null;
    let activeDocumentName = null;
    let activeDocumentSize = null;

    // DOM Elements
    const uploadZone = document.getElementById("upload-zone");
    const fileInput = document.getElementById("file-input");
    const progressContainer = document.getElementById("upload-progress-container");
    const progressFill = document.getElementById("upload-progress-fill");
    const progressStatus = document.getElementById("upload-progress-status");
    
    const activeDocCard = document.getElementById("active-doc-card");
    const emptyDocState = document.getElementById("empty-doc-state");
    const docMetaState = document.getElementById("doc-meta-state");
    const activeDocName = document.getElementById("active-doc-name");
    const activeDocSize = document.getElementById("active-doc-size");
    const activeDocUuid = document.getElementById("active-doc-uuid");
    const removeDocBtn = document.getElementById("remove-doc-btn");

    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const chatMessages = document.getElementById("chat-messages");
    const sendButton = document.getElementById("send-button");

    const traceContainer = document.getElementById("trace-container");
    const emptyTraceState = document.getElementById("empty-trace-state");

    const emptyStructuredState = document.getElementById("empty-structured-state");
    const structuredOutputDisplay = document.getElementById("structured-output-display");
    const structuredMetadataGrid = document.getElementById("structured-metadata-grid");
    const structuredBullets = document.getElementById("structured-bullets");
    const structuredJsonRaw = document.getElementById("structured-json-raw");

    // File Upload Event Listeners
    uploadZone.addEventListener("click", () => fileInput.click());
    
    uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadZone.classList.add("dragover");
    });

    uploadZone.addEventListener("dragleave", () => {
        uploadZone.classList.remove("dragover");
    });

    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    removeDocBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        clearActiveDocument();
    });

    // File Upload Processing
    async function handleFileUpload(file) {
        // Simple client validations
        const allowedExtensions = [".pdf", ".png", ".jpg", ".jpeg"];
        const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
        if (!allowedExtensions.includes(ext)) {
            appendSystemMessage("Error: Only PDF and image files are supported.", "system-error");
            return;
        }

        const maxLimit = 10 * 1024 * 1024; // 10MB
        if (file.size > maxLimit) {
            appendSystemMessage("Error: File exceeds the 10MB size limit.", "system-error");
            return;
        }

        // Show progress UI
        progressContainer.style.display = "block";
        progressFill.style.width = "0%";
        progressStatus.textContent = "Preparing file...";

        const formData = new FormData();
        formData.append("file", file);

        try {
            progressFill.style.width = "40%";
            progressStatus.textContent = "Uploading document...";

            const response = await fetch("/process-document", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Upload failed");
            }

            progressFill.style.width = "80%";
            progressStatus.textContent = "Parsing and indexing text...";

            const data = await response.json();
            
            progressFill.style.width = "100%";
            progressStatus.textContent = "Complete!";
            
            setTimeout(() => {
                progressContainer.style.display = "none";
            }, 1000);

            // Set active state
            activeDocumentId = data.document_id;
            activeDocumentName = data.filename;
            activeDocumentSize = formatBytes(file.size);

            renderActiveDocument();
            appendSystemMessage(`Document successfully processed: "${file.name}". ${data.message}`, "assistant");

        } catch (error) {
            console.error("Upload failed:", error);
            progressContainer.style.display = "none";
            appendSystemMessage(`Upload failed: ${error.message}`, "system-error");
        }
    }

    function renderActiveDocument() {
        emptyDocState.style.display = "none";
        docMetaState.style.display = "flex";
        activeDocName.textContent = activeDocumentName;
        activeDocSize.textContent = activeDocumentSize;
        activeDocUuid.textContent = activeDocumentId;
    }

    function clearActiveDocument() {
        activeDocumentId = null;
        activeDocumentName = null;
        activeDocumentSize = null;
        docMetaState.style.display = "none";
        emptyDocState.style.display = "block";
        fileInput.value = "";
        appendSystemMessage("Active document context cleared from workspace.", "assistant");
    }

    // Chat Submission
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = chatInput.value.trim();
        if (!text) return;

        // Reset inputs and UI
        chatInput.value = "";
        appendMessage(text, "user");
        
        // Clear old trace steps and structured displays for the new query
        traceContainer.replaceChildren();
        emptyTraceState.style.display = "none";

        // Disable send button while active
        sendButton.disabled = true;

        try {
            const response = await fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    message: text,
                    document_id: activeDocumentId
                })
            });

            if (!response.ok) {
                throw new Error(`Server returned status ${response.status}`);
            }

            // Stream reader initialization
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                
                // Keep the final incomplete line in buffer
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const rawData = line.slice(6).trim();
                        if (rawData) {
                            try {
                                const event = JSON.parse(rawData);
                                processSSEEvent(event);
                            } catch (e) {
                                console.error("Error parsing event stream json line:", e);
                            }
                        }
                    }
                }
            }

        } catch (err) {
            console.error("Streaming error:", err);
            appendSystemMessage(`Streaming connection failed: ${err.message}`, "system-error");
        } finally {
            sendButton.disabled = false;
        }
    });

    // Process single SSE event item
    function processSSEEvent(data) {
        const { event, node, content } = data;

        if (event === "node_start" && node) {
            createOrUpdateTraceStep(node, "active", `Agent ${node} is executing...`);
        } 
        else if (event === "node_end" && node) {
            createOrUpdateTraceStep(node, "completed", `Agent ${node} finished execution.`);
        } 
        else if (event === "agent_message" && node) {
            // Map graph node names to pretty agent titles
            const nodeTitles = {
                "supervisor": "Supervisor Agent",
                "ocr_worker": "OCR Extraction Agent",
                "rag_worker": "RAG Retrieval Agent",
                "structured_worker": "Structured Output Agent"
            };
            const title = nodeTitles[node] || node;
            appendMessage(content, "assistant", title);
            createOrUpdateTraceStep(node, "completed", content);
        }
        else if (event === "final_answer" && data.data) {
            renderStructuredData(data.data);
            createOrUpdateTraceStep(node || "structured_worker", "completed", "Structured JSON data created.");
        }
        else if (event === "error") {
            appendSystemMessage(content, "system-error");
        }
    }

    // Trace list manager
    function createOrUpdateTraceStep(nodeId, status, detailsText) {
        // Check if step element already exists
        let stepEl = document.getElementById(`trace-step-${nodeId}`);
        if (!stepEl) {
            stepEl = document.createElement("div");
            stepEl.id = `trace-step-${nodeId}`;
            stepEl.classList.add("trace-step", nodeId);

            const header = document.createElement("div");
            header.classList.add("trace-step-header");

            const title = document.createElement("span");
            title.classList.add("trace-node-name");
            title.textContent = nodeId.replace("_", " ");

            const time = document.createElement("span");
            time.classList.add("trace-step-time");
            time.textContent = new Date().toLocaleTimeString();

            header.appendChild(title);
            header.appendChild(time);
            stepEl.appendChild(header);

            const desc = document.createElement("p");
            desc.classList.add("trace-step-desc");
            stepEl.appendChild(desc);

            traceContainer.appendChild(stepEl);
        }

        // Update step status class list
        if (status === "active") {
            stepEl.classList.add("active");
        } else {
            stepEl.classList.remove("active");
        }

        // Set detailed text safely
        const descEl = stepEl.querySelector(".trace-step-desc");
        if (descEl) {
            descEl.textContent = detailsText;
        }

        // Auto-scroll trace container
        traceContainer.scrollTop = traceContainer.scrollHeight;
    }

    // Render Pydantic Structured JSON Output
    function renderStructuredData(schemaData) {
        emptyStructuredState.style.display = "none";
        structuredOutputDisplay.style.display = "flex";

        // 1. Clear previous content safely
        structuredMetadataGrid.replaceChildren();
        structuredBullets.replaceChildren();
        structuredJsonRaw.textContent = "";

        // 2. Render core items in grid
        const typeItem = createMetaItem("Document Type", schemaData.document_type || "Unknown");
        const summaryItem = createMetaItem("Executive Summary", schemaData.summary || "No summary provided.");
        structuredMetadataGrid.appendChild(typeItem);
        structuredMetadataGrid.appendChild(summaryItem);

        // 3. Render extra dynamic metadata
        if (schemaData.extracted_metadata) {
            for (const [key, val] of Object.entries(schemaData.extracted_metadata)) {
                // Formatting key
                const formattedKey = key.replace(/_/g, " ");
                const metaItem = createMetaItem(formattedKey, val);
                structuredMetadataGrid.appendChild(metaItem);
            }
        }

        // 4. Render bullet takeaways
        if (schemaData.key_takeaways && schemaData.key_takeaways.length > 0) {
            schemaData.key_takeaways.forEach(point => {
                const li = document.createElement("li");
                li.textContent = point;
                structuredBullets.appendChild(li);
            });
        } else {
            const li = document.createElement("li");
            li.textContent = "No bullet takeaways extracted.";
            structuredBullets.appendChild(li);
        }

        // 5. Populate Raw JSON text
        structuredJsonRaw.textContent = JSON.stringify(schemaData, null, 2);
    }

    function createMetaItem(labelStr, valStr) {
        const div = document.createElement("div");
        div.classList.add("meta-item");
        
        const label = document.createElement("span");
        label.classList.add("meta-label");
        label.textContent = labelStr;
        
        const val = document.createElement("span");
        val.classList.add("meta-val");
        val.textContent = valStr;
        val.title = valStr;

        div.appendChild(label);
        div.appendChild(val);
        return div;
    }

    // DOM Helpers
    function appendMessage(text, sender, agentTitle = null) {
        const bubble = document.createElement("div");
        bubble.classList.add("message", sender);

        if (agentTitle) {
            const titleSpan = document.createElement("strong");
            titleSpan.style.display = "block";
            titleSpan.style.marginBottom = "6px";
            titleSpan.style.fontSize = "11px";
            titleSpan.style.color = "#a5b4fc";
            titleSpan.textContent = agentTitle;
            bubble.appendChild(titleSpan);
        }

        // Render line breaks as breaks or wrap in paragraph tags
        const textSpan = document.createElement("span");
        textSpan.textContent = text;
        textSpan.style.whiteSpace = "pre-wrap";
        bubble.appendChild(textSpan);

        chatMessages.appendChild(bubble);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendSystemMessage(text, type) {
        const bubble = document.createElement("div");
        bubble.classList.add("message", type);
        
        const textSpan = document.createElement("span");
        textSpan.textContent = text;
        bubble.appendChild(textSpan);

        chatMessages.appendChild(bubble);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function formatBytes(bytes, decimals = 2) {
        if (!+bytes) return "0 Bytes";
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ["Bytes", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
    }
})();
