// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
  anchor.addEventListener("click", function (e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute("href"));
    if (target) {
      target.scrollIntoView({ behavior: "smooth" });
    }
  });
});

// Dynamic title animation on hover
const title = document.querySelector("h1");
if (title) {
  title.addEventListener("mouseenter", () => {
    title.style.transform = "scale(1.02)";
    title.style.transition = "transform 0.3s ease";
  });
  title.addEventListener("mouseleave", () => {
    title.style.transform = "scale(1)";
  });
}
