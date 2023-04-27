document$.subscribe(() => {
	// MathJax.typesetPromise()
	if (typeof katex !== "undefined") {
		var maths = document.querySelectorAll('.arithmatex'),
			tex;

		const lazyLoadOptions = {
			threshold: 0,
			rootMargin: "0px 0px 600px 0px"
		};

		const mathObserver = new IntersectionObserver((entries, mathObserver) => {
			entries.forEach((entry) => {
				if (!entry.isIntersecting) return

				tex = entry.textContent || entry.innerText;
				if (tex.startsWith('\\(') && tex.endsWith('\\)')) {
					katex.render(tex.slice(2, -2), entry, { 'displayMode': false });
				} else if (tex.startsWith('\\[') && tex.endsWith('\\]')) {
					katex.render(tex.slice(2, -2), entry, { 'displayMode': true });
				}
			});
		}, lazyLoadOptions)
		
		maths.forEach((math) => {
			mathObserver.observe(math);
			math.onload = function () {
				asset.classList.add("loaded")
			}
		})
	}
})

// load event gets fired only once; this script gets called every time

window.addEventListener("load", function () {
	quicklink_options = {
		threshold: 0.25, // default = 0, % in decimal that needs to be visible
		ignores: [
			/\/api\/?/, // all "/api/*" pathnames
			uri => uri.includes('.zip'), // all ".zip" extensions
			uri => uri.includes('#'), // prefetches to URLs containing URL fragment
			(uri, elem) => elem.hasAttribute('noprefetch') // all <a> tags with "noprefetch" attribute
		] // String, RegExp, or Function values
	}

	quicklink.listen(quicklink_options);
	const next_page = document.querySelector(".md-footer__link--next").href.toLowerCase();
	quicklink.prefetch(
		[next_page],
		true // priority; false means low priority
	);
});

// render mathjax
// window.MathJax = {
// 	// i want to load these manually :(
// 	loader: {
// 		load: [
// 			'[tex]/color',
// 			'[tex]/cases',
// 			'[tex]/mhchem',
// 		]
// 	},
// 	tex: {
// 		packages: {
// 			'[+]': [
// 				'color',
// 				'cases',
// 				'mhchem',
// 				'textmacros'
// 			]
// 		},
// 		inlineMath: [
// 			["\\(", "\\)"],
// 			["$", "$"]
// 		],
// 		displayMath: [
// 			["\\[", "\\]"],
// 			["$$", "$$"]
// 		],
// 		processEscapes: true,
// 		processEnvironments: true
// 	},
// 	options: {
// 		ignoreHtmlClass: ".*|",
// 		processHtmlClass: "arithmatex"
// 	}
// };
