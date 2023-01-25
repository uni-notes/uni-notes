window.MathJax = {
	// i want to load these manually :(
	loader: {
		load: [
			'[tex]/color',
			'[tex]/cases',
			'[tex]/mhchem',
			'[tex]/textmacros'
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
		tags: 'ams',
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