import { usePageFrontmatter } from "@vuepress/client";
import { computed, defineComponent, h, resolveComponent, } from "vue";
import { RenderDefault, hasGlobalComponent } from "vuepress-shared/client";
import BreadCrumb from "@theme-hope/components/BreadCrumb";
// import MarkdownContent from "@theme-hope/components/MarkdownContent";
import PageNav from "@theme-hope/components/PageNav";
import PageTitle from "@theme-hope/components/PageTitle";
// import { useThemeLocaleData } from "@theme-hope/composables/index";
import PageMeta from "@theme-hope/modules/info/components/PageMeta";
// import TOC from "@theme-hope/modules/info/components/TOC";
import { useDarkmode } from "@theme-hope/modules/outlook/composables/index";
import ArticleList from "@theme-hope/modules/blog/components/ArticleList";
// import { useArticles } from "@theme-hope/modules/blog/composables/index";
import DropTransition from "@theme-hope/components/transitions/DropTransition";
// import InfoPanel from "@theme-hope/modules/blog/components/InfoPanel";
import { useCategoryMap } from "@theme-hope/modules/blog/composables/index";

// import "../styles/page.scss";
export default defineComponent({
    name: "MenuPage",
    setup(_props, { slots }) {
        const { isDarkmode } = useDarkmode();
        const categoryMap = useCategoryMap();
        const frontmatter = usePageFrontmatter();

        const items = computed(() => {
            return categoryMap.value.map[frontmatter.value.categoryid].items ?? []

        });

        return () => h("main", { class: "page", id: "main-content" }, h(hasGlobalComponent("LocalEncrypt")
            ? resolveComponent("LocalEncrypt")
            : RenderDefault, () => [
            slots["top"]?.(),

            h(BreadCrumb),
            h(PageTitle),
            slots["contentBefore"]?.(),

            h("main", { class: "menu-page", id: "category-menu-content" }, [
                h(DropTransition, { appear: true, delay: 0.24 }, () => h(ArticleList, { items: items.value })),
            ]),
            // h(MarkdownContent),
            slots["contentAfter"]?.(),

            h(PageMeta),
            h(PageNav),
            hasGlobalComponent("CommentService")
                ? h(resolveComponent("CommentService"), {
                    darkmode: isDarkmode.value,
                })
                : null,
            slots["bottom"]?.(),
        ]));
    },
});
//# sourceMappingURL=MenuPage.js.map