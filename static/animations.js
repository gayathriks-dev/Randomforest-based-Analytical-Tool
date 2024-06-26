// static/animations.js

document.addEventListener("DOMContentLoaded", function () {
    gsap.from(".hero-content", { opacity: 0, duration: 1, y: -50, ease: "power4.out" });
});
