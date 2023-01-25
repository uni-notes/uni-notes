window.MathJax = {
	// i want to load these manually :(
	loader: {
		load: [
			'[tex]/color',
			'[tex]/cases',
			'[tex]/mhchem',
		]
	},
	tex: {
		packages: {
			'[+]': [
				'color',
				'cases',
				'mhchem',
				'textmacros'
			]
		},
		inlineMath: [
			["\\(", "\\)"],
			["$", "$"]
		],
		displayMath: [
			["\\[", "\\]"],
			["$$", "$$"]
		],
		processEscapes: true,
		processEnvironments: true
	},
	options: {
		ignoreHtmlClass: ".*|",
		processHtmlClass: "arithmatex"
	}
};

document$.subscribe(() => {
	MathJax.typesetPromise()
})