document.addEventListener("DOMContentLoaded", () => {
  // === DOM Elements ===
  const form = document.getElementById("generator-form");
  const tabButtons = document.querySelectorAll(".tab-btn");
  const modeInput = document.getElementById("mode-input");
  const styleSection = document.getElementById("style-section");
  const faceSection = document.getElementById("face-section");
  const faceImageUpload = document.getElementById("face-image-upload");
  const faceImagePreview = document.getElementById("face-image-preview");
  const fileNameDisplay = document.getElementById("file-name-display");
  const uploadArea = document.getElementById("upload-area");
  const generateBtn = document.getElementById("generate-btn");
  const outputImage = document.getElementById("output-image");
  const placeholderText = document.getElementById("placeholder-text");
  const errorMessage = document.getElementById("error-message");
  const downloadBtn = document.getElementById("download-btn");
  const shareBtn = document.getElementById("share-btn");

  // === Event Listeners ===
  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => handleTabChange(btn));
  });

  faceImageUpload.addEventListener("change", handleFileUpload);

  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("drag-over");
  });

  uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("drag-over");
  });

  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("drag-over");

    if (e.dataTransfer.files.length) {
      faceImageUpload.files = e.dataTransfer.files;
      handleFileUpload();
    }
  });

  generateBtn.addEventListener("click", handleGenerate);

  // === Functions ===
  function handleTabChange(clickedTab) {
    tabButtons.forEach((tab) => tab.classList.remove("active"));
    clickedTab.classList.add("active");
    const mode = clickedTab.dataset.mode;
    modeInput.value = mode;

    styleSection.classList.toggle(
      "active",
      mode === "text_style" || mode === "face_style"
    );
    faceSection.classList.toggle("active", mode === "face_style");

    // Hide face preview and file name if not in face_style mode
    if (mode !== "face_style") {
      faceImagePreview.style.display = "none";
      fileNameDisplay.textContent = "No file chosen";
      faceImageUpload.value = "";
    }

    resetOutput();
  }

  function handleFileUpload() {
    const file = faceImageUpload.files[0];
    if (file) {
      fileNameDisplay.textContent = file.name;
      const reader = new FileReader();
      reader.onload = (e) => {
        faceImagePreview.src = e.target.result;
        faceImagePreview.style.display = "block";
      };
      reader.readAsDataURL(file);
      resetOutput(); // Reset output if new image uploaded
    } else {
      fileNameDisplay.textContent = "No file chosen";
      faceImagePreview.style.display = "none";
    }
  }

  async function handleGenerate() {
    const mode = modeInput.value;
    const promptInput = document.getElementById("prompt");
    const prompt = promptInput.value.trim();
    const negativePrompt = document
      .getElementById("negative_prompt")
      .value.trim();
    const styleSelect = document.getElementById("style-select");
    const selectedStyle = styleSelect ? styleSelect.value : null;
    const faceFile = faceImageUpload.files[0];

    // --- Validation ---
    if (!prompt) {
      showError("Please describe your emoji");
      promptInput.focus();
      return;
    }
    if (mode === "face_style" && !faceFile) {
      showError("Please upload a face image");
      return;
    }
    if ((mode === "face_style" || mode === "text_style") && !selectedStyle) {
      showError("Please select a style");
      return;
    }

    // --- Prepare Data and UI for API Call ---
    setLoading(true);
    resetOutput();

    const formData = new FormData();
    formData.append("mode", mode);
    formData.append("prompt", prompt);
    if (negativePrompt) {
      formData.append("negative_prompt", negativePrompt);
    }
    if (selectedStyle && (mode === "text_style" || mode === "face_style")) {
      formData.append("style", selectedStyle);
    }
    if (faceFile && mode === "face_style") {
      formData.append("face_image", faceFile);
    }

    // --- API Call ---
    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        body: formData,
        // headers: { 'Accept': 'application/json' } // Not needed for FormData usually
      });

      const result = await response.json();

      if (!response.ok) {
        // Handle HTTP errors (4xx, 5xx) with error message from JSON payload
        throw new Error(result.error || `HTTP error ${response.status}`);
      }

      if (result.image_data) {
        // Success: Display image
        outputImage.src = result.image_data;
        outputImage.style.display = "block";
        placeholderText.style.display = "none";
        downloadBtn.disabled = false;
        shareBtn.disabled = false; // Enable share button
      } else if (result.error) {
        // Handle errors reported in JSON payload even with 200 OK (less common)
        showError(result.error);
      } else {
        // Handle unexpected success response format
        showError("Received an unexpected response from the server.");
      }
    } catch (error) {
      console.error("Generation API Error:", error);
      showError(error.message || "Network error or server unavailable.");
    } finally {
      // --- Reset UI State ---
      setLoading(false);
    }
  }

  function setLoading(isLoading) {
    if (isLoading) {
      generateBtn.classList.add("loading");
      generateBtn.disabled = true;
    } else {
      generateBtn.classList.remove("loading");
      generateBtn.disabled = false;
    }
  }

  function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = "block";
    // Hide output image and show placeholder on error
    outputImage.style.display = "none";
    placeholderText.style.display = "flex";
    setLoading(false); // Ensure button is re-enabled
  }

  function resetOutput() {
    outputImage.style.display = "none";
    outputImage.src = "#"; // Clear src
    placeholderText.style.display = "flex";
    errorMessage.style.display = "none";
    errorMessage.textContent = "";
    downloadBtn.disabled = true;
    shareBtn.disabled = true;
  }

  // === Download and Share functionality ===
  downloadBtn.addEventListener("click", () => {
    if (outputImage.src && outputImage.src !== "#") {
      const link = document.createElement("a");
      link.href = outputImage.src;
      // Generate a more descriptive filename maybe
      const mode = modeInput.value;
      const promptStart = document
        .getElementById("prompt")
        .value.trim()
        .substring(0, 15)
        .replace(/\s+/g, "_");
      link.download = `MojiSan_${mode}_${promptStart || "emoji"}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  });

  shareBtn.addEventListener("click", async () => {
    if (outputImage.src && outputImage.src !== "#" && navigator.share) {
      try {
        // Convert data URL to blob for sharing
        const response = await fetch(outputImage.src);
        const blob = await response.blob();
        const file = new File([blob], "synthemoji.png", { type: "image/png" });

        await navigator.share({
          title: "My AI Emoji!",
          text: "Check out this emoji I created with MojiSan Studio!",
          files: [file],
        });
        console.log("Successful share");
      } catch (error) {
        console.error("Share failed:", error);
        alert("Sharing failed or was cancelled.");
      }
    } else if (navigator.share) {
      alert("Generate an emoji first to share it!");
    } else {
      alert(
        "Web Share API not supported in your browser. Try downloading instead."
      );
    }
  });

  // === Initialize ===
  resetOutput(); // Set initial output state
  // Set initial visibility based on default checked radio
  handleTabChange(document.querySelector(".tab-btn.active") || tabButtons[0]);
});
