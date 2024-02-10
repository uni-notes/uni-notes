document$.subscribe(() => {
	// MathJax.typesetPromise()
	if (typeof katex !== "undefined") {
		var maths = document.querySelectorAll('.arithmatex'),
			tex;

		const no_of_screens_lookahead = 1;
		const lazyLoadOptions = {
			threshold: 0,
			rootMargin: `0px 0px ${no_of_screens_lookahead * window.innerHeight}px 0px`
		};

		const mathObserver = new IntersectionObserver((entries, observer) => {
			entries.forEach((entry) => {
				if (!entry.isIntersecting) return

				const math = entry.target
				
				tex = math.textContent || math.innerText;

				if (tex.startsWith('\\(') && tex.endsWith('\\)')) {
					katex.render(tex.slice(2, -2), math, { 'displayMode': false });
				} else if (tex.startsWith('\\[') && tex.endsWith('\\]')) {
					katex.render(tex.slice(2, -2), math, { 'displayMode': true });
				}
				math.classList.add("loaded");
				observer.unobserve(math);
			});
		}, lazyLoadOptions)

		maths.forEach((math) => {
			mathObserver.observe(math);
		})
	}
})

document$.subscribe(() => {
	// To move metadata
	var firstH1 = document.querySelector("h1");
	var metadata = document.querySelector(".md-source-file");

	if (metadata && firstH1) {
		firstH1.parentNode.insertBefore(metadata, firstH1.nextSibling);
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
