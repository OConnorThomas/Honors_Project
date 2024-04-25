
const colorPicker = document.getElementById('colorPicker');

function update_c1() {
  const selectedColor = this.value;
  document.documentElement.style.setProperty('--user-color-1', selectedColor);
}

colorPicker.addEventListener('input', update_c1());
