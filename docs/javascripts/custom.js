window.addEventListener("load", function () {
  options = {
    threshold: 0.25, // default = 0, % in decimal that needs to be visible
    ignores: [
      /\/api\/?/, // all "/api/*" pathnames
      uri => uri.includes('.zip'), // all ".zip" extensions
      uri => uri.includes('#'), // prefetches to URLs containing URL fragment
      (uri, elem) => elem.hasAttribute('noprefetch') // all <a> tags with "noprefetch" attribute
    ] // String, RegExp, or Function values
  }

  quicklink.listen(options);
  const next_page = document.querySelector(".md-footer__link--next").href.toLowerCase();
  quicklink.prefetch(
    [next_page],
    true // priority; false means low priority
  );
});
