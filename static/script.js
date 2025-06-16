let cambios = [];

document.getElementById("imagen").addEventListener("change", async function () {
  const file = this.files[0];
  if (!file) return;

  document.getElementById("preview").src = URL.createObjectURL(file);
  document.getElementById("preview").style.display = "block";
  document.getElementById("resultado").src = "";
  document.getElementById("descargar").style.display = "none";
  document.getElementById("formulario-cambio").style.display = "none";
  document.getElementById("colores").style.display = "none";
  document.getElementById("opciones").style.display = "block";

  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("/detectar-colores", {
    method: "POST",
    body: formData
  });

  const data = await res.json();
  mostrarColores(data.colores);
});

function mostrarColores(colores) {
  const contenedor = document.getElementById("colores");
  contenedor.innerHTML = "";
  contenedor.style.display = "flex";

  colores.forEach((color, idx) => {
    const box = document.createElement("div");
    box.className = "color-box";
    box.style.backgroundColor = color;

    const colorInput = document.createElement("input");
    colorInput.type = "color";
    colorInput.value = color;
    colorInput.oninput = () => guardarCambio(idx, colorInput.value);

    box.appendChild(colorInput);
    contenedor.appendChild(box);
  });

  document.getElementById("formulario-cambio").style.display = "block";
}

function guardarCambio(cluster, nuevoColor) {
  const existente = cambios.find(c => c.cluster === cluster);
  if (existente) {
    existente.color = nuevoColor;
  } else {
    cambios.push({ cluster, color: nuevoColor });
  }
}

async function enviarCambios() {
  if (cambios.length === 0) {
    alert("No hiciste ningún cambio.");
    return;
  }

  const quitarFondo = document.getElementById("quitar_fondo").checked;
  const preservarBlancos = document.getElementById("preservar_blancos").checked;

  const loader = document.getElementById("loader");
  loader.style.display = "block";

  const formData = new FormData();
  formData.append("recolores", JSON.stringify(cambios));
  formData.append("quitar_fondo", quitarFondo);
  formData.append("preservar_blancos", preservarBlancos);

  const res = await fetch("/cambiar-colores-multiples", {
    method: "POST",
    body: formData
  });

  const data = await res.json();

  loader.style.display = "none";
  document.getElementById("resultado").src = data.url;
  document.getElementById("link-descarga").href = data.url;
  document.getElementById("descargar").style.display = "block";
}
