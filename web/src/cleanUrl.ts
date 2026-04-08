if (typeof window !== "undefined") {
  const clean = setInterval(() => {
    if (window.location.href.endsWith("#") || window.location.href.endsWith("/#")) {
      window.history.replaceState(null, "", window.location.pathname || "/");
    }
  }, 200);
  setTimeout(() => clearInterval(clean), 15000);
}
