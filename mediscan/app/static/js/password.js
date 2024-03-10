function validatePasswordsDoc() {
    var password = document.getElementById("password").value;
    var confirmPassword = document.getElementById("confirmPassword").value;
    var message = document.getElementById("message");
  
    if (password === confirmPassword && password !== "") {
      message.innerHTML = "Passwords match!";
      message.style.color = "green"; // Change color to green
    } else {
      message.innerHTML = "Passwords do not match or empty!";
      message.style.color = "red"; // Change color to red
    }
  }
  