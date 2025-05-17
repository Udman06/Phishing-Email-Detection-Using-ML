async function checkEmail() {
    const emailText = document.getElementById("rawdata").value;
    const response = await fetch("http://127.0.0.1:5000", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ email: emailText }),
    });
  
    const result = await response.json();
    document.getElementById("result").innerText = result.prediction;
  }