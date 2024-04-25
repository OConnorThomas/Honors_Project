let logOutBtn = document.getElementById("logOutBtn");
let welcomeUser = document.getElementById("userName");
let userName = localStorage.getItem("userName");

function get_logged_in() {
  return localStorage.getItem("_LOGGEDIN");
}

function updateContent() {
  if (get_logged_in() == 1) {
    welcomeUser.textContent = userName;
  } else {
    welcomeUser.textContent = "NULLuser";
  }
}

updateContent();

logOutBtn.addEventListener("click", function () {
  localStorage.setItem("_LOGGEDIN", 0);
  window.location.href = "home.html";
});
