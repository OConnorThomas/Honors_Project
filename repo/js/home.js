let accounttype = document.getElementById("accountbox")

function get_logged_in() {
  return localStorage.getItem("_LOGGEDIN");
}

function updateContent() {
  if (get_logged_in() == 1) {
    accounttype.insertAdjacentHTML("afterend", "<a href=\"account.html\"><img src=\"repo/images/Generic-Profile-Image.png\"alt=\"Profile Picture\" class=\"const-size\"/></a>");
    accounttype.insertAdjacentHTML("afterend", "<a href=\"settings.html\"><img src=\"repo/images/gear.png\" alt=\"Settings button\" class=\"const-size settings-icon-size\"/></a>")
  } else {
    accounttype.insertAdjacentHTML("afterend", "<div class=\"pad\"><ul><li><a href=\"signup.html\" class=\"shadowed-text invert-wb\">Signup</a></li></ul></div>");
    accounttype.insertAdjacentHTML("afterend", "<div class=\"pad\"><ul><li><a href=\"login.html\" class=\"shadowed-text invert-wb\">Login</a></li></ul></div>");
  }
}

updateContent();
