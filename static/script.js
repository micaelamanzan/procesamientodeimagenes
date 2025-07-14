// Variable para almacenar los cambios de color seleccionados por el usuario
let cambios = [];
let currentSessionId = null;
let selectedColorPreviewElements = new Map();
let cambiosAplicados = false;
const imagenInput = document.getElementById("imagen");
const previewImg = document.getElementById("preview");
const previewPlaceholder = document.getElementById("preview-placeholder");
const resultadoImg = document.getElementById("resultado");
const descargarLink = document.getElementById("link-descarga");
const descargarPanel = document.getElementById("descargar-buttons");
const coloresContainer = document.getElementById("colores");
const opcionesPanel = document.getElementById("opciones");
const colorPanel = document.getElementById("color-panel");
const loader = document.getElementById("loader");
const resultadoPanel = document.getElementById("resultado-panel");
const errorMessageDiv = document.getElementById("error-message");
const aplicarCambiosBtn = document.getElementById("aplicar-cambios-btn-final");
const reloadButton = document.getElementById("reload-button");
const aplicarCambiosContainer = document.getElementById("aplicar-cambios-container");

function showErrorMessage(message) {
  if (errorMessageDiv) {
    errorMessageDiv.textContent = message;
    errorMessageDiv.classList.remove("hidden");
  }
}

function hideErrorMessage() {
  if (errorMessageDiv) {
    errorMessageDiv.textContent = "";
    errorMessageDiv.classList.add("hidden");
  }
}

function toggleLoader(show) {
  if (loader) loader.classList.toggle("hidden", !show);
  if (aplicarCambiosBtn) {
    aplicarCambiosBtn.disabled = show;
    aplicarCambiosBtn.textContent = show ? "Procesando..." : "Aplicar cambios";
  }
}

function resetUI() {
  previewImg.src = "";
  previewImg.classList.add("hidden");
  previewPlaceholder.classList.remove("hidden");
  resultadoImg.src = "";
  resultadoImg.classList.add("hidden");
  descargarPanel.classList.add("hidden");
  aplicarCambiosContainer.classList.add("hidden");
  coloresContainer.innerHTML = "";
  coloresContainer.classList.add("hidden");
  opcionesPanel.classList.add("hidden");
  colorPanel.classList.add("hidden");
  resultadoPanel.classList.add("hidden");
  toggleLoader(false);
  hideErrorMessage();
  cambios = [];
  currentSessionId = null;
  cambiosAplicados = false;
  selectedColorPreviewElements.clear();
  if (aplicarCambiosBtn) {
    aplicarCambiosBtn.disabled = false;
    aplicarCambiosBtn.textContent = "Aplicar cambios";
    aplicarCambiosBtn.classList.remove("bg-gray-400", "cursor-not-allowed");
    aplicarCambiosBtn.classList.add("bg-green-600", "hover:bg-green-700");

    // Restaurar listener en caso de reemplazo
    aplicarCambiosBtn.replaceWith(aplicarCambiosBtn.cloneNode(true));
    const nuevoBtn = document.getElementById("aplicar-cambios-btn-final");
    nuevoBtn.addEventListener("click", enviarCambios);
  }
  document.getElementById("quitar_fondo").checked = false;
  document.getElementById("preservar_blancos").checked = false;
  imagenInput.value = null;
}

imagenInput.addEventListener("change", async function () {
  const file = this.files[0];
  if (!file) {
    resetUI();
    return;
  }
  resetUI();
  previewImg.src = URL.createObjectURL(file);
  previewImg.classList.remove("hidden");
  previewPlaceholder.classList.add("hidden");
  opcionesPanel.classList.remove("hidden");
  toggleLoader(true);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/detectar-colores", { method: "POST", body: formData });
    if (!res.ok) throw new Error((await res.json()).detail || "Error al detectar colores.");
    const data = await res.json();
    currentSessionId = data.session_id;
    mostrarColores(data.colores);
  } catch (error) {
    showErrorMessage(`Error al procesar la imagen: ${error.message}`);
    resetUI();
  } finally {
    toggleLoader(false);
  }
});

function mostrarColores(colores) {
  coloresContainer.innerHTML = "";
  coloresContainer.classList.remove("hidden");
  colorPanel.classList.remove("hidden");
  colores.forEach((color, idx) => {
    const box = document.createElement("div");
    box.className = "color-box flex flex-col items-center justify-center p-1";
    box.style.backgroundColor = color;

    const originalColorLabel = document.createElement("div");
    originalColorLabel.className = "w-full h-1/2 rounded-t-md";
    originalColorLabel.style.backgroundColor = color;
    originalColorLabel.title = `Original: ${color}`;
    box.appendChild(originalColorLabel);

    const selectedColorPreview = document.createElement("div");
    selectedColorPreview.className = "w-full h-1/2 rounded-b-md border-t border-gray-300";
    selectedColorPreview.style.backgroundColor = color;
    selectedColorPreview.title = `Seleccionado: ${color}`;
    box.appendChild(selectedColorPreview);

    selectedColorPreviewElements.set(idx, selectedColorPreview);

    const colorInput = document.createElement("input");
    colorInput.type = "color";
    colorInput.value = color;
    colorInput.oninput = () => guardarCambio(idx, colorInput.value);

    box.appendChild(colorInput);
    coloresContainer.appendChild(box);
  });
  aplicarCambiosContainer.classList.remove("hidden");
}

function guardarCambio(clusterId, nuevoColor) {
  const existente = cambios.find(c => c.cluster === clusterId);
  if (existente) {
    existente.color = nuevoColor;
  } else {
    cambios.push({ cluster: clusterId, color: nuevoColor });
  }
  const previewElement = selectedColorPreviewElements.get(clusterId);
  if (previewElement) {
    previewElement.style.backgroundColor = nuevoColor;
    previewElement.title = `Seleccionado: ${nuevoColor}`;
  }
  hideErrorMessage();
}

async function enviarCambios() {
  if (cambiosAplicados) return;

  const quitarFondo = document.getElementById("quitar_fondo").checked;
  const preservarBlancos = document.getElementById("preservar_blancos").checked;

  if (cambios.length === 0 && !quitarFondo && !preservarBlancos) {
    showErrorMessage("No has realizado ningún cambio de color ni seleccionado opciones de fondo.");
    return;
  }

  if (!currentSessionId || typeof currentSessionId !== "string") {
    showErrorMessage("No hay una sesión de imagen activa o es inválida. Por favor, sube una imagen nuevamente.");
    return;
  }

  toggleLoader(true);

  const formData = new FormData();
  formData.append("session_id", currentSessionId);
  formData.append("recolores", JSON.stringify(cambios));
  formData.append("quitar_fondo", quitarFondo);
  formData.append("preservar_blancos", preservarBlancos);

  try {
    const res = await fetch("/cambiar-colores-multiples", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const errorData = await res.json();
      throw new Error(errorData.detail || "Error al aplicar cambios.");
    }

    const data = await res.json();
    resultadoImg.src = data.url;
    resultadoImg.classList.remove("hidden");
    descargarLink.href = data.url;
    descargarPanel.classList.remove("hidden");
    resultadoPanel.classList.remove("hidden");
    hideErrorMessage();

    if (aplicarCambiosBtn) {
      aplicarCambiosBtn.disabled = true;
      aplicarCambiosBtn.textContent = "Cambios aplicados";
      aplicarCambiosBtn.classList.remove("bg-green-600", "hover:bg-green-700");
      aplicarCambiosBtn.classList.add("bg-gray-400", "cursor-not-allowed");
      cambiosAplicados = true;
    }
  } catch (error) {
    showErrorMessage(`Error al aplicar los cambios: ${error.message}`);
    resultadoImg.classList.add("hidden");
    descargarPanel.classList.add("hidden");
  } finally {
    toggleLoader(false);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  if (aplicarCambiosBtn) {
    aplicarCambiosBtn.addEventListener('click', enviarCambios);
  }
  if (reloadButton) {
    reloadButton.addEventListener('click', resetUI);
  }
});
