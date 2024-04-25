// Dynamic Dark Mode
function updateBackgroundColor() {
  const hour = new Date().getHours();
  const isDaytime = hour > 6 && hour < 18;
  
  // Set background color to a lighter shade during the day and a darker shade during the night
  const backgroundColor = isDaytime ? '#e6e6e6' : '#222222'; // Light gray for day, darker gray for night
  const textColor = isDaytime ? '#000000' : '#ffffff';
  const sectionColor = isDaytime ? '#ffffff' : '#111111';
  document.documentElement.style.setProperty('--section-color', sectionColor);
  document.documentElement.style.setProperty('--background-color', backgroundColor);
  document.documentElement.style.setProperty('--text-color', textColor);
}
// Dynamic Dark Mode

// Listen for keydown events on the entire document
document.addEventListener('keydown', function(event) {
  // Check if the Escape key is pressed
  if (event.key === "Escape") {
      // Call the closePopup function
      closePopup();
  }
});

// Function to open the popup
function openPopup(src, altText) {
  // Get the modal
  var modal = document.getElementById("popup-modal");
  // Get the image and insert it inside the modal
  var modalImg = document.getElementById("popup-image");
  var captionText = document.getElementById("caption");
  modal.style.display = "block";
  modalImg.src = src;
  // Set the alt text as the caption
  captionText.innerHTML = altText;
}

// Function to close the popup
function closePopup() {
  var modal = document.getElementById("popup-modal");
  modal.style.display = "none";
}

// Initial call
updateBackgroundColor();

// Set interval to update background color every minute (60000 milliseconds)
setInterval(updateBackgroundColor, 60000);